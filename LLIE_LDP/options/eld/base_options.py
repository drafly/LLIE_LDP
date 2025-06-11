from options.base_option import BaseOptions as Base
from util import util
import os
import torch
import numpy as np
import random


class BaseOptions(Base):
    def initialize(self):
        Base.initialize(self)
        self.parser.add_argument('--iter_num', type=int, default=2)
        self.parser.add_argument('--netG', type=str, default='unet', help='chooses which architecture to use for netG.')
        self.parser.add_argument('--resid_net', type=str, default='resid_nafnet')
        self.parser.add_argument('--GFMs_net', type=str, default='GFM')
        self.parser.add_argument('--adaptive_res_and_x0', action="store_true", help='adaptively combine the clean image and the image removed noise')
        self.parser.add_argument('--with_photon', action="store_true")
        self.parser.add_argument('--concat_origin', action="store_true")
        
        self.parser.add_argument('--resid', action="store_true", help='predict the noise instead of the clean image')
        self.parser.add_argument('--channels', '-c', type=int, default=4, help='in/out channels (4: bayer; 9: xtrans')
        self.parser.add_argument('--stage_in', type=str, default='raw', help='input stage [raw|srgb]')
        self.parser.add_argument('--stage_out', type=str, default='raw', help='output stage [raw|srgb]')
        self.parser.add_argument('--stage_eval', type=str, default='raw', help='output stage [raw|srgb]')
        self.parser.add_argument('--model_path', type=str, default=None, help='model checkpoint to use.')
        self.parser.add_argument('--resid_model_path', type=str, default=None, help='model checkpoint to use.')
        self.parser.add_argument('--include', type=int, default=None, help='select camera in ELD dataset')
        self.parser.add_argument('--gt_wb', action='store_true', help='use white balance of ground truth')
        self.parser.add_argument('--finetune', action='store_true', help='use white balance of ground truth')
        self.parser.add_argument('--crf', action='store_true', help='use CRF to render sRGB images')
        self.parser.add_argument('--epoch', type=int, default=300)
        
        self.initialized = True
        self.isTrain = False

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        torch.backends.cudnn.deterministic = True
        torch.manual_seed(self.opt.seed)
        np.random.seed(self.opt.seed) # seed for every module
        random.seed(self.opt.seed)

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        if not self.opt.no_verbose:
            print('------------ Options -------------')
            for k, v in sorted(args.items()):
                print('%s: %s' % (str(k), str(v)))
            print('-------------- End ----------------')

        # save to the disk
        self.opt.name = self.opt.name or '_'.join([self.opt.model])
        # self.opt.resid_name = self.opt.resid_name or '_'.join([self.opt.model])
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        # expr_dir2 = os.path.join(self.opt.checkpoints_dir, self.opt.resid_name)
        util.mkdirs(expr_dir)
        # util.mkdirs(expr_dir2)
        file_name = os.path.join(expr_dir, 'opt.txt')
        # file_name = os.path.join(expr_dir2, 'opt.txt')

        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')

        if self.opt.debug:
            self.opt.display_freq = 20
            self.opt.print_freq = 20
            self.opt.nEpochs = 40
            self.opt.max_dataset_size = 100
            self.opt.no_log = False
            self.opt.nThreads = 0
            self.opt.decay_iter = 0
            self.opt.serial_batches = True
            self.opt.no_flip = True
        
        return self.opt
