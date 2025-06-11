import torch
from torch import nn
import torch.nn.functional as F
from torchvision.utils import save_image
import os
import numpy as np
import fnmatch
from collections import OrderedDict

import util.util as util
import util.index as index
import models.networks as networks
from models import arch, losses

from .base_model import BaseModel
from PIL import Image
from os.path import join

import rawpy
import util.process as process
from torchvision.utils import save_image
from models.ELD_model import ELDModelBase, IlluminanceCorrect, tensor2im


class ELDModelIter(ELDModelBase):
    def __init__(self):
        self.epoch = 0
        self.iterations = 0
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.corrector = IlluminanceCorrect()
        self.CRF = None

    def print_network(self):
        print('---------------------Resid Model ---------------------')
        networks.print_network(self.resid_net)
        print('--------------------- Model ---------------------')
        networks.print_network(self.netG)

    def _eval(self):
        self.resid_net.eval()
        self.netG.eval()

    def _train(self):
        self.resid_net.train()
        self.netG.train()

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if vars(opt).get("adaptive_loss", False):
            # self.weight = nn.Parameter(torch.zeros(size=(opt.iter_num, )))
            self.weight = nn.Parameter(torch.ones(size=(opt.iter_num, ))*(1/(self.opt.iter_num+1)))
            # self.resid_weight = nn.Parameter(torch.ones(size=(opt.iter_num, ))*(1/(self.opt.iter_num+1)))
            self.resid_weight = nn.Parameter(torch.tensor(1.0))
        else:
            self.weight = None
            self.resid_weight = None

        # init CRF function
        if opt.crf:
            self.CRF = process.load_CRF()

        if opt.stage_in == 'raw':
            in_channels = opt.channels
        elif opt.stage_in == 'srgb':
            in_channels = 3
        else:
            raise NotImplementedError(
                'Invalid Input Stage: {}'.format(opt.stage_in))

        if opt.stage_out == 'raw':
            out_channels = opt.channels
        elif opt.stage_out == 'srgb':
            out_channels = 3
        else:
            raise NotImplementedError(
                'Invalid Output Stage: {}'.format(opt.stage_in))


        self.netG = arch.__dict__[self.opt.netG](in_channels, out_channels, resid=opt.resid, with_photon=opt.with_photon,
                                                 concat_origin=opt.concat_origin, adaptive_res_and_x0=opt.adaptive_res_and_x0).to(self.device)

        self.resid_net = arch.__dict__[self.opt.resid_net](in_channels, out_channels).to(self.device)
        self.GFMs_net = arch.__dict__[self.opt.GFMs_net](in_channels, feature_num=2, bias=True, padding_mode='reflect').to(self.device)

        if self.isTrain:
            # define loss functions
            self.loss_dic = losses.init_loss(opt)

            # initialize optimizers
            
            # param_list = list(self.netG.parameters())
            param_list = list(self.netG.parameters()) + list(self.GFMs_net.parameters())
            resid_param_list = list(self.resid_net.parameters())
            if opt.adaptive_loss:
                param_list.append(self.weight)
                resid_param_list.append(self.resid_weight)
            self.optimizer_G = torch.optim.Adam(param_list,
                                                lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.wd)

            self.resid_optimizer_G = torch.optim.Adam(resid_param_list,
                                                lr=opt.resid_lr, betas=(opt.resid_beta1, 0.999), weight_decay=opt.resid_wd)

            self._init_optimizer([self.optimizer_G])
            self.resid_init_optimizer([self.resid_optimizer_G])


        if opt.finetune:
            self.finetune(self)

        if opt.resume:
            self.load(self, opt.resume_epoch)


        if opt.no_verbose is False:
            self.print_network()

    def backward_resid(self, resid_output=None):
        self.loss_resid = None


        target = torch.clamp(self.target, 0, 1)

        resid_target = (target - self.input)


        self.loss_resid = self.loss_dic['pixel_resid'].get_loss(
            resid_output, resid_target)

        self.loss_resid.backward(retain_graph=True)

    def backward_G(self):
        self.loss_G = 0
        self.loss_pixel = None

        for idx, (output, metadata) in enumerate(self.output_list):
            self.loss_pixel = self.loss_dic['pixel'].get_loss(
                output, self.target)
            if self.opt.auxloss:
                subloss = ((metadata["out"] -self.target).abs()*metadata["mask"])+((metadata["out_by_resid"] -self.target).abs()*(1-metadata["mask"]))
                self.loss_pixel += subloss.mean()
            # self.loss_pixel.backward()
            if self.weight is None:
                self.loss_G += self.loss_pixel
            else:
                self.loss_G += self.loss_pixel * (self.weight[idx].clip(0,1) if idx < self.opt.iter_num else 1-self.weight.sum())



        self.loss_G.backward()

    def resid_forward(self):
        input = self.input
        ratio, K = int(self.ratio), float(self.K)
        photon =  input / ratio / K * 15583
        resid_net_input = torch.cat([input, photon / 15583], dim=1)
        resid_output, metadata = self.resid_net(resid_net_input)
        return resid_output

    def forward(self, iter_num=0, resid_output=None):
        input_i = self.input

        self.resid_output = resid_output

        def to_photon(x, ratio, K):
            # x: 0-1 image
            return x/ratio/K*15583

        def to_image(x, ratio_next, target_ratio, K):
            return x/ratio_next*target_ratio*K/15583

        def _forward(input_i):
            net_input = input_i 
            metadata = None
            current_phothon = to_photon(input_i, ratio/ratio_current, K)
            if self.opt.with_photon:
                net_input = torch.cat([net_input, current_phothon/15583], dim=1)
            if self.opt.concat_origin:
                resid_output_i =self.GFMs_net(self.input, self.resid_output)
                net_input = torch.cat([net_input, resid_output_i], dim=1)


            if self.opt.chop:
                output = self.forward_chop(net_input)
            else:
                output, metadata = self.netG(net_input)
            return output, metadata
        
        # sorted(set(np.linspace(1,ratio,iter_num).astype(int)))
        if self.netG.training:
            bs=len(self.ratio)
            if bs>1:
                ratio, K = self.ratio.tolist(), self.K.to(input_i.device).view(bs,1,1,1).float()
                ratio_list=torch.tensor([sorted(
                    set([1]+list(np.random.choice(list(range(2, max(int(r),     iter_num+2))), iter_num, replace=False))+[r])) for r in ratio])
                ratio_list = ratio_list.to(input_i.device).view(bs,-1,1,1,1).float()
                ratio=torch.tensor(ratio).to(input_i.device).view(-1,1,1,1).float()
                # raise NotImplementedError("batch size > 1 is still not supported")
            else:
                ratio, K = int(self.ratio), float(self.K)

                if ratio <= 1:
                    ratio_list = [1, ratio]
                else:
                    sample_range = list(range(1, ratio))
                    replace_flag = iter_num > len(sample_range)
                    ratio_list = sorted(
                        set([1] + list(np.random.choice(sample_range, iter_num, replace=replace_flag)) + [ratio]))
        else:
            ratio, K = int(self.ratio), float(self.K)
            if iter_num==0:
                ratio_list=[1, ratio] # [1,100]
            else:
                # ratio_list=[1,50,100,200]
                ratio_list = np.linspace(1,ratio,iter_num+2)
                # ratio_list = [1, 100, 200, 300]
                # ratio_list = [1, 33, 66, 100]
                # ratio_list=[1, 20, 40 , 60]#, 80] #  
                # ratio_list = [1,2,4,8]
            # ratio_list = np.linspace(1,100,iter_num+2)

        self.output_list = []
        iter_num=len(ratio_list)-1 if not isinstance(ratio_list, torch.Tensor) else ratio_list.shape[1]-1
        for i in range(iter_num):
            if isinstance(ratio_list, torch.Tensor):
                ratio_current, ratio_next = ratio_list[:,i], ratio_list[:,i+1]
            else: 
                ratio_current, ratio_next = ratio_list[i], ratio_list[i+1]
            output_list = list(_forward(input_i))
            # input_i = to_image(to_photon(input_i.clamp(0, 1), ratio/ratio_current, K) + torch.poisson(
            #     to_photon(output.clamp(0, 1), ratio/(ratio_next-ratio_current), K)), ratio_next, ratio, K)
            output_list[0] = self.corrector(output_list[0], self.target)
            output = output_list[0]
            self.output_end = output_list[0]

            if ratio_next <= 1:
                incre_mean = to_photon(output.clamp(0, 1), ratio / (ratio_next), K)
                assert (incre_mean >= 0).all(), "Error: incre_mean contains negative values!"
                incre = (torch.poisson(incre_mean) - incre_mean).detach() + incre_mean
            else:
                incre_mean = to_photon(output.clamp(0, 1), ratio / (ratio_next - 1), K)
                assert (incre_mean >= 0).all(), "Error: incre_mean contains negative values!"
                incre = (torch.poisson(incre_mean) - incre_mean).detach() + incre_mean
            
            input_i = to_image(to_photon(self.input.clamp(0, 1), ratio, K) + incre, ratio_next, ratio, K)
            # self.input = to_image(to_photon(self.input, ratio, self.K)*ratio_current + to_photon(output, ratio, self.K)*(ratio_next-ratio_current), ratio_next, ratio, self.K)
            self.output_list.append(output_list)

        # self.output = _forward(input_i)
        # self.output_list.append(self.output)
        return self.output_list[-1]

    def forward_chop(self, x, base=16):
        b, c, h, w = x.size()
        h_half, w_half = h // 2, w // 2

        shave_h = np.ceil(h_half / base) * base - h_half
        shave_w = np.ceil(w_half / base) * base - w_half

        shave_h = shave_h if shave_h >= 10 else shave_h + base
        shave_w = shave_w if shave_w >= 10 else shave_w + base

        h_size, w_size = int(h_half + shave_h), int(w_half + shave_w)

        inputs = [
            x[:, :, 0:h_size, 0:w_size],
            x[:, :, 0:h_size, (w - w_size):w],
            x[:, :, (h - h_size):h, 0:w_size],
            x[:, :, (h - h_size):h, (w - w_size):w]
        ]

        outputs = [self.netG(input_i)[0] for input_i in inputs]

        c = outputs[0].shape[1]
        output = x.new(b, c, h, w)

        output[:, :, 0:h_half, 0:w_half] \
            = outputs[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] \
            = outputs[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, h_half:h, 0:w_half] \
            = outputs[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] \
            = outputs[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

        return output

    def optimize_parameters(self):
        self._train()
        resid_output = self.resid_forward()

        self.forward(iter_num=self.opt.iter_num, resid_output=resid_output)

        self.resid_optimizer_G.zero_grad()
        self.optimizer_G.zero_grad()
        self.backward_resid(resid_output=resid_output)

        self.backward_G()
        self.resid_optimizer_G.step()
        self.optimizer_G.step()

    def get_current_errors(self):
        ret_errors = OrderedDict()
        if self.loss_pixel is not None:
            ret_errors['Pixel'] = self.loss_pixel.item()
            if self.weight is not None:
                ret_errors['w1'] = self.weight.detach()[0].item()
                ret_errors['w2'] = self.weight.detach()[1].item()
        return ret_errors

    def get_current_visuals(self):
        ret_visuals = OrderedDict()
        ret_visuals['input'] = tensor2im(
            self.input, visualize=True).astype(np.uint8)
        ret_visuals['output'] = tensor2im(
            self.output, visualize=True).astype(np.uint8)
        ret_visuals['target'] = tensor2im(
            self.target, visualize=True).astype(np.uint8)

        return ret_visuals

    @staticmethod
    def load(model, resume_epoch=None):
        model_path = model.opt.model_path
        resid_model_path = model.opt.resid_model_path
        state_dict = None
        resid_state_dict = None

        if model_path is None:
            model_path = util.get_model_list(
                model.save_dir, 'model', epoch=resume_epoch)
        if resid_model_path is None:
            resid_model_path = util.get_model_list(model.resid_save_dir, 'resid_model')
        state_dict = torch.load(model_path)
        model.epoch = state_dict['epoch']
        model.iterations = state_dict['iterations']
        model.netG.load_state_dict(state_dict['netG'])
        model.GFMs_net.load_state_dict(state_dict['GFMs_net'])
        resid_state_dict = torch.load(resid_model_path)
        model.resid_net.load_state_dict(resid_state_dict['resid_net'])
        if model.isTrain:
            model.optimizer_G.load_state_dict(state_dict['opt_g'])
            model.resid_optimizer_G.load_state_dict(resid_state_dict['resid_opt_g'])
        # else:
        #     state_dict = torch.load(model_path)
        #     model.netG.load_state_dict(state_dict['netG'])
        #     model.epoch = state_dict['epoch']
        #     model.iterations = state_dict['iterations']
        #     if model.isTrain:
        #         model.optimizer_G.load_state_dict(state_dict['opt_g'])

        print('Resume from epoch %d, iteration %d' %
              (model.epoch, model.iterations))
        return state_dict, resid_state_dict

    @staticmethod
    def finetune(model):
        model_path = model.opt.model_path
        resid_model_path = model.opt.resid_model_path
        state_dict = None
        resid_state_dict = None

        state_dict = torch.load(model_path)
        model.iterations = state_dict['iterations']
        model.netG.load_state_dict(state_dict['netG'])
        model.GFMs_net.load_state_dict(state_dict['GFMs_net'])
        resid_state_dict = torch.load(resid_model_path)
        model.resid_net.load_state_dict(resid_state_dict['resid_net'])
        return state_dict, resid_state_dict

    def state_dict(self):
        state_dict = {
            'netG': self.netG.state_dict(),
            'GFMs_net': self.GFMs_net.state_dict(),
            'opt_g': self.optimizer_G.state_dict(),
            'epoch': self.epoch, 'iterations': self.iterations
        }

        return state_dict
    def resid_state_dict(self):
        resid_state_dict = {
            'resid_net': self.resid_net.state_dict(),
            'resid_opt_g': self.resid_optimizer_G.state_dict(),
            'epoch': self.epoch, 'iterations': self.iterations
        }
        return resid_state_dict

