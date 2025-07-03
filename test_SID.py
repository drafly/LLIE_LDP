from options.eld.base_options import BaseOptions 
from engine import Engine
import torch
import torch.backends.cudnn as cudnn
import dataset.sid_dataset as datasets
import dataset
import noise
import numpy as np

opt = BaseOptions().parse()

cudnn.benchmark = True

datadir = '/root/autodl-tmp/SID'

expo_ratio = [100, 250, 300] # [100, 250, 300] # , 100, 250
# expo_ratio = [100,250,300]
read_expo_ratio = lambda x: float(x.split('_')[-1][:-5])

train_fns = dataset.read_paired_fns('./dataset/Sony_train.txt')
eval_fns = dataset.read_paired_fns('./dataset/Sony_val.txt')
test_fns = dataset.read_paired_fns('./dataset/Sony_test.txt')

eval_fns_list = [[fn for fn in eval_fns if min(int(read_expo_ratio(fn[1])/read_expo_ratio(fn[0])), 300)==ratio] for ratio in expo_ratio]
test_fns_list = [[fn for fn in test_fns if min(int(read_expo_ratio(fn[1])/read_expo_ratio(fn[0])), 300)==ratio] for ratio in expo_ratio]
eval_fns_list = [lst_1 + lst_2 for lst_1, lst_2 in zip(eval_fns_list, test_fns_list)]

# evaluate 15 indoor scenes (but you can also evaluate the performance on the whole dataset)
indoor_ids = dataset.read_paired_fns('./SID_Sony_15_paired.txt')
eval_fns_list = [[(fn[0], fn[1]) for fn in indoor_ids if int(fn[2]) == ratio] for ratio in expo_ratio]
# eval_fns_list = [[fn for fn in indoor_ids if min(int(read_expo_ratio(fn[1])/read_expo_ratio(fn[0])), 300)==ratio] for ratio in expo_ratio]

cameras = ['CanonEOS5D4', 'CanonEOS70D', 'CanonEOS700D', 'NikonD850', 'SonyA7S2']
noise_model = noise.NoiseModel(model="P+g", include=4)
eval_datasets = [datasets.SIDDataset(datadir, fns, noise_model, memorize=False, size=None, augment=False, stage_in=opt.stage_in, stage_out=opt.stage_out) for fns in eval_fns_list]
# eval_datasets = [datasets.SIDDataset(datadir, fns, noise_model, memorize=False, size=None, augment=False, stage_in=opt.stage_in, stage_out=opt.stage_out) for fns in test_fns_list ]

eval_dataloaders = [torch.utils.data.DataLoader(
    eval_dataset, batch_size=1, shuffle=False,
    num_workers=opt.nThreads, pin_memory=True) for eval_dataset in eval_datasets]


"""Main Loop"""
engine = Engine(opt)

for ratio, dataloader in zip(expo_ratio, eval_dataloaders):
    # if ratio not in [300]: continue
    print('Eval ratio {}'.format(ratio))


    res = engine.eval(dataloader, dataset_name='sid_eval_{}'.format(ratio), correct=True, crop=True, iter_num=opt.iter_num, savedir=f"images/{opt.model_path.split('/')[-2]}/{ratio}")




