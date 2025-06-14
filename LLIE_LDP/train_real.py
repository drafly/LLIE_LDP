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
import noise

opt = TrainOptions().parse()

cudnn.benchmark = True

evaldir = '/root/autodl-tmp/Sony'
traindir = '/root/autodl-tmp/train'

expo_ratio = [100, 250, 300]
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


CRF = None
if opt.crf:
    print('[i] enable CRF')
    CRF = process.load_CRF()

# ['CanonEOS5D4', 'CanonEOS70D', 'CanonEOS700D', 'NikonD850', 'SonyA7S2']     
noise_model = noise.NoiseModel(model="P+G+r+u", include=4)

if opt.stage_in == 'srgb':
    if opt.crf:
        input_data = LMDBDataset(join(traindir, 'SID_Sony_input_SRGB_CRF.db'))
    else:
        input_data = LMDBDataset(join(traindir, 'SID_Sony_input_SRGB.db'))
else:
    input_data = LMDBDataset(join(traindir, 'SID_Sony_input_Raw.db'), noise_model=noise_model)

if opt.stage_out == 'srgb':
    if opt.crf:
        target_data = LMDBDataset(join(traindir, 'SID_Sony_target_SRGB_CRF.db'))
    else:
        target_data = LMDBDataset(join(traindir, 'SID_Sony_target_SRGB.db'))
else:
    target_data = LMDBDataset(join(traindir, 'SID_Sony_target_Raw.db'))


train_dataset =  datasets.ELDTrainDataset(target_dataset=target_data, input_datasets=[input_data], syn_noise=opt.syn_noise)

eval_datasets = [datasets.SIDDataset(evaldir, fns, noise_maker= noise_model, size=None, memorize=False, augment=False, stage_in=opt.stage_in, stage_out=opt.stage_out, gt_wb=opt.gt_wb, CRF=CRF) for fns in eval_fns_list]


train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=opt.batchSize, shuffle=True,
    num_workers=opt.nThreads, pin_memory=True, worker_init_fn=worker_init_fn)


eval_dataloaders = [torch.utils.data.DataLoader(
    eval_dataset, batch_size=1, shuffle=False,
    num_workers=0, pin_memory=True) for eval_dataset in eval_datasets]


"""Main Loop"""
engine = Engine(opt)

engine.model.opt.save_epoch_freq = 100

engine.set_learning_rate(1e-4)
engine.set_resid_learning_rate(1e-4)

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
    
    engine.train(train_dataloader)

    train_dataset.reset()

    if engine.epoch % 10 == 0:
        try:
            print("Eval sid 100:")
            engine.eval(eval_dataloaders[0], dataset_name='sid_eval_100', correct=True, iter_num=opt.iter_num)
            print("Eval sid 250:")
            engine.eval(eval_dataloaders[1], dataset_name='sid_eval_250', correct=True, iter_num=opt.iter_num)
            print("Eval sid 300:")
            engine.eval(eval_dataloaders[2], dataset_name='sid_eval_300', correct=True, iter_num=opt.iter_num)
        except:
            pass
