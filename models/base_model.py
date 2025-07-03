import os
import torch
import util.util as util


class BaseModel():
    def name(self):
        return self.__class__.__name__.lower()

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.resid_save_dir = os.path.join(opt.checkpoints_dir, opt.resid_name or 'residnet_naf1')
        self._count = 0

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def resid_forward(self):
        pass


    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def resid_update_learning_rate(self):
        for scheduler in self.resid_schedulers:
            scheduler.step()
        resid_lr = self.resid_optimizers[0].param_groups[0]['resid_lr']
        print('residnet learning rate = %.7f' % resid_lr)


    def print_optimizer_param(self):
        print(self.optimizers[-1])

    def print_resid_optimizer_param(self):
        print(self.resid_optimizers[-1])
        

    def save(self, label=None):
        epoch = self.epoch
        iterations = self.iterations

        if label is None:
            model_name = os.path.join(self.save_dir, 'model' + '_%03d_%08d.pt' % ((epoch), (iterations)))
            resid_model_name = os.path.join(self.resid_save_dir, 'resid_model' + '_%03d_%08d.pt' % ((epoch), (iterations)))
        else:
            model_name = os.path.join(self.save_dir, 'model' + '_' + label + '.pt')
            resid_model_name = os.path.join(self.resid_save_dir, 'resid_model' + '_' + label + '.pt')

        torch.save(self.state_dict(), model_name)
        torch.save(self.resid_state_dict(), resid_model_name)


    def _init_optimizer(self, optimizers):
        self.optimizers = optimizers
        # reinitilize schedulers
        self.schedulers = []
        for optimizer in self.optimizers:
            util.set_opt_param(optimizer, 'initial_lr', self.opt.lr)
            util.set_opt_param(optimizer, 'weight_decay', self.opt.wd)

    def resid_init_optimizer(self, resid_optimizers):
        self.resid_optimizers = resid_optimizers
        # reinitilize schedulers
        self.resid_schedulers = []
        for optimizer in self.resid_optimizers:
            util.set_opt_param(optimizer, 'initial_resid_lr', self.opt.resid_lr)
            util.set_opt_param(optimizer, 'resid_weight_decay', self.opt.resid_wd)
