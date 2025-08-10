from options.eld.base_options import BaseOptions
from engine import Engine
import torch
import torch.backends.cudnn as cudnn
import dataset.sid_dataset as datasets
import dataset
import noise_lrd
import numpy as np

opt = BaseOptions().parse()

cudnn.benchmark = True

evaldir = '/root/autodl-tmp/LRD'

EV = [-1, -2, -3]  


EVs_ids = dataset.read_paired_fns('/root/autodl-tmp/LRD/LRD_test_list.txt')

eval_fns_list = [
    [(fn[0], fn[1], fn[2]) for fn in EVs_ids if int(fn[3].replace('EV', '')) == ev]
    for ev in EV
]

noise_model = noise_lrd.NoiseModel(model="P+g", include=4)

eval_datasets = [datasets.SIDDataset(evaldir, fns, noise_maker=noise_model, size=None, memorize=False, augment=False,
                                     stage_in=opt.stage_in, stage_out=opt.stage_out, gt_wb=opt.gt_wb, CRF=CRF) for fns
                 in eval_fns_list]

eval_dataloaders = [torch.utils.data.DataLoader(
    eval_dataset, batch_size=1, shuffle=False,
    num_workers=opt.nThreads, pin_memory=True) for eval_dataset in eval_datasets]

"""Main Loop"""
engine = Engine(opt)

for ev, dataloader in zip(EV, eval_dataloaders):

    print('Eval ratio {}'.format(ev))

    res = engine.eval(dataloader, dataset_name='lrd_test_{}'.format(ev), correct=True, crop=True,
                      iter_num=opt.iter_num, savedir=f"images/{opt.model_path.split('/')[-2]}/{ev}")


