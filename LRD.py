from os.path import join
from options.eld.train_options import TrainOptions
from engine import Engine
import torch
import torch.backends.cudnn as cudnn
import dataset.sid_dataset as datasets
import util.util as util
import dataset
import numpy as np
from dataset.sid_dataset import worker_init_fn
from dataset.lmdb_dataset import LMDBDataset
from util import process
import noise_lrd
import dataset.lmdb_dataset as lmdb_dataset

opt = TrainOptions().parse()

cudnn.benchmark = True

evaldir = '/root/autodl-tmp/LRD'
traindir = '/root/autodl-tmp/train'

EV = [-1, -2, -3]  # [100, 300]


EVs_ids = dataset.read_paired_fns('/root/autodl-tmp/LRD/LRD_test_list.txt')

eval_fns_list = [
    [(fn[0], fn[1], fn[2]) for fn in EVs_ids if int(fn[3].replace('EV', '')) == ev]
    for ev in EV
]

CRF = None
if opt.crf:
    print('[i] enable CRF')
    CRF = process.load_CRF()

repeat = 1 if opt.max_dataset_size is None else 1288 / opt.max_dataset_size
print('[i] repeat:', repeat)

noise_model = noise_lrd.NoiseModel(model="P+g", include=4)

if opt.stage_in == 'srgb':
    if opt.crf:
        input_data = LMDBDataset(join(traindir, 'SID_Sony_input_SRGB_CRF.db'))
    else:
        input_data = LMDBDataset(join(traindir, 'SID_Sony_input_SRGB.db'))
else:
    input_data = datasets.SynDataset(
        lmdb_dataset.LMDBDataset(join(traindir, 'LDR_target_Raw.db')),
        noise_maker=noise_model, num_burst=1,
        size=opt.max_dataset_size, repeat=repeat, continuous_noise=False)

if opt.stage_out == 'srgb':
    if opt.crf:
        target_data = LMDBDataset(join(traindir, 'SID_Sony_target_SRGB_CRF.db'))
    else:
        target_data = LMDBDataset(join(traindir, 'SID_Sony_target_SRGB.db'))
else:
    target_data = lmdb_dataset.LMDBDataset(
        join(traindir, 'LDR_target_Raw.db'),
        size=opt.max_dataset_size, repeat=repeat)

train_dataset = datasets.ELDTrainDataset(target_dataset=target_data, input_datasets=[input_data],
                                         syn_noise=opt.syn_noise)

eval_datasets = [datasets.SIDDataset(evaldir, fns, noise_maker=noise_model, size=None, memorize=False, augment=False,
                                     stage_in=opt.stage_in, stage_out=opt.stage_out, gt_wb=opt.gt_wb, CRF=CRF) for fns
                 in eval_fns_list]

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=opt.batchSize, shuffle=True,
    num_workers=opt.nThreads, pin_memory=True, worker_init_fn=worker_init_fn)

eval_dataloaders = [torch.utils.data.DataLoader(
    eval_dataset, batch_size=1, shuffle=False,
    num_workers=0, pin_memory=True) for eval_dataset in eval_datasets]

"""Main Loop"""
engine = Engine(opt)

engine.model.opt.save_epoch_freq = 100

engine.set_learning_rate(1e-5)
engine.set_resid_learning_rate(1e-5)

while engine.epoch < opt.epoch:
    np.random.seed()
    if engine.epoch == opt.epoch // 2:  # 150
        engine.set_learning_rate(opt.lr / 2)
    if engine.epoch == 100:
        engine.set_resid_learning_rate(5e-5)
    if engine.epoch == int(opt.epoch * 0.9):  # 270
        engine.set_learning_rate(opt.lr / 10)
    if engine.epoch == 180:
        engine.set_resid_learning_rate(1e-5)

    engine.train_finetune(train_dataloader)


    if engine.epoch % 10 == 0:
        try:
            print("Eval sid -1EV:")
            engine.eval(eval_dataloaders[0], dataset_name='sid_eval_-1EV', correct=True, iter_num=opt.iter_num)
            print("Eval sid -2EV:")
            engine.eval(eval_dataloaders[1], dataset_name='sid_eval_-2EV', correct=True, iter_num=opt.iter_num)
            print("Eval sid -3EV:")
            engine.eval(eval_dataloaders[2], dataset_name='sid_eval_-3EV', correct=True, iter_num=opt.iter_num)
        except:
            pass
