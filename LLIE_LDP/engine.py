import numpy as np
import torch
import util.util as util
import models
import time
import os
import sys
from os.path import join
from torchvision.utils import save_image



class Engine(object):
    def __init__(self, opt):
        self.opt = opt
        self.writer = None
        self.model = None
        self.best_val_loss = 1e6
        self.__setup()

        self.best_scores = {}


    def __setup(self):
        self.basedir = join('checkpoints', self.opt.name)
        if not os.path.exists(self.basedir):
            os.mkdir(self.basedir)
        
        opt = self.opt
        
        """Model"""
        self.model = models.__dict__[self.opt.model]()
        self.model.initialize(opt)
        if not opt.no_log:
            self.writer = util.get_summary_writer(os.path.join(self.basedir, 'logs'))

    def train(self, train_loader, **kwargs):
        print('\nEpoch: %d' % self.epoch)
        avg_meters = util.AverageMeters()
        opt = self.opt
        model = self.model
        epoch = self.epoch

        epoch_start_time = time.time()
        # model.print_optimizer_param()
        for i, data in enumerate(train_loader):
            
            iter_start_time = time.time()
            iterations = self.iterations

            model.set_input(data, mode='train')
            model.optimize_parameters(**kwargs)
            
            errors = model.get_current_errors()
            avg_meters.update(errors)
            util.progress_bar(i, len(train_loader), str(avg_meters))

            if not opt.no_log:
                util.write_loss(self.writer, 'train', avg_meters, iterations, log_to_file=True)

            self.iterations += 1
    
        self.epoch += 1

        if not self.opt.no_log:
            if self.epoch % opt.save_epoch_freq == 0:
                print('saving the model at epoch %d, iters %d' %
                    (self.epoch, self.iterations))
                model.save()
            
            print('saving the latest model at the end of epoch %d, iters %d' % 
                (self.epoch, self.iterations))
            model.save(label='latest')

            print('Time Taken: %d sec' %
                (time.time() - epoch_start_time))
                
        model.update_learning_rate()
        model.resid_update_learning_rate()
        # train_loader.reset()

    def train_finetune(self, train_loader, **kwargs):
        print('\nEpoch: %d' % self.epoch)
        avg_meters = util.AverageMeters()
        opt = self.opt
        model = self.model
        epoch = self.epoch

        epoch_start_time = time.time()
        # model.print_optimizer_param()

        for i, data in enumerate(train_loader):

            iter_start_time = time.time()
            iterations = self.iterations

            model.set_input(data, mode='train')

            model.optimize_parameters(**kwargs)

            errors = model.get_current_errors()
            avg_meters.update(errors)
            util.progress_bar(i, len(train_loader), str(avg_meters))

            if not opt.no_log:
                util.write_loss(self.writer, 'train', avg_meters, iterations, log_to_file=True)

            self.iterations += 1

        self.epoch += 1

        if not self.opt.no_log:
            if self.epoch % opt.save_epoch_freq == 0:
                print('saving the model at epoch %d, iters %d' %
                      (self.epoch, self.iterations))
                model.save()

            print('saving the latest model at the end of epoch %d, iters %d' %
                  (self.epoch, self.iterations))
            model.save(label='latest')

            print('Time Taken: %d sec' %
                  (time.time() - epoch_start_time))

        model.update_learning_rate()
        model.resid_update_learning_rate()
        # train_loader.reset()

    def eval(self, val_loader, dataset_name, savedir=None, loss_key=None,
             **kwargs):
        iter_num = kwargs.get("iter_num", None)
        if iter_num:
            print("[i] Evaluation using iterartion number of %d"%iter_num)
        avg_meters = util.AverageMeters()
        model = self.model
        opt = self.opt
        if dataset_name not in self.best_scores:
            self.best_scores[dataset_name] = {
                'PSNR': float('-inf'),
                'SSIM': float('-inf')
            }

        with torch.no_grad():

            for i, data in enumerate(val_loader):

                index = model.eval(i, data, savedir=savedir, **kwargs)


                avg_meters.update(index)

                
                util.progress_bar(i, len(val_loader), str(avg_meters))



        if not opt.no_log:
            util.write_loss(self.writer, join('eval', dataset_name), avg_meters, self.epoch, log_to_file=True)
        
        if loss_key is not None:
            val_loss = avg_meters[loss_key]
            if val_loss < self.best_val_loss: # larger value indicates better
                self.best_val_loss = val_loss
                print('saving the best model at the end of epoch %d, iters %d' % 
                    (self.epoch, self.iterations))
                model.save(label='best_{}_{}'.format(loss_key, dataset_name))
        '''
        current_psnr = avg_meters['PSNR']
        current_ssnr = avg_meters['SSIM']
       
        if (current_psnr > self.best_scores[dataset_name]['PSNR'] and
            current_ssnr > self.best_scores[dataset_name]['SSIM']):
            self.best_scores[dataset_name]['PSNR'] = current_psnr
            self.best_scores[dataset_name]['SSIM'] = current_ssnr
            print('Saving best model for %s at epoch %d, iters %d' %
                  (dataset_name, self.epoch, self.iterations))
            model.save(label='best_psnr_ssim_{}'.format(dataset_name))
        '''

        return avg_meters

    def test(self, test_loader, savedir=None, **kwargs):
        model = self.model
        opt = self.opt
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                model.test(data, savedir=savedir, **kwargs)
                util.progress_bar(i, len(test_loader))

    def set_learning_rate(self, lr):
        for optimizer in self.model.optimizers:
            print('[i] set learning rate to {}'.format(lr))
            util.set_opt_param(optimizer, 'lr', lr)

    def set_resid_learning_rate(self, resid_lr):
        for optimizer in self.model.resid_optimizers:
            print('[i] set learning rate to {}'.format(resid_lr))
            util.set_opt_param(optimizer, 'resid_lr', resid_lr)

    @property
    def iterations(self):
        return self.model.iterations

    @iterations.setter
    def iterations(self, i):
        self.model.iterations = i

    @property
    def epoch(self):
        return self.model.epoch

    @epoch.setter
    def epoch(self, e):
        self.model.epoch = e
