# See in the Dark (SID) dataset
import torch
import os
import rawpy
import numpy as np
from os.path import join

from matplotlib import pyplot as plt

import dataset.torchdata as torchdata
import util.process as process
from util.util import loadmat
import exifread
import pickle
import random
from PIL import Image
from torchvision.utils import save_image 

BaseDataset = torchdata.Dataset


def tensor2im(image_tensor, visualize=False, video=False):

    # 检查输入类型，如果是 numpy.ndarray 就不做 cpu 和 detach 操作
    if isinstance(image_tensor, torch.Tensor):
        image_tensor = image_tensor.detach().cpu()  # 只对 tensor 调用 .cpu() 和 .detach()
    elif isinstance(image_tensor, np.ndarray):
        # 如果是 numpy.ndarray，直接跳过这些操作
        pass
    else:
        raise TypeError("Input should be a torch.Tensor or a numpy.ndarray")

    if visualize:
        image_tensor = image_tensor[:, 0:3, ...]

    if not video:
        if isinstance(image_tensor, torch.Tensor):
            image_numpy = image_tensor[0].float().numpy()
        else:
            image_numpy = image_tensor[0].astype(float)  # 如果是 numpy.ndarray，直接使用
        # print("//////////////////////////////")
        # 在调用 transpose 之前，打印 image_numpy 的形状
        # print(f"image_numpy shape before transpose: {image_numpy.shape}")

        # 检查形状并根据需要调整 transpose 操作
        if len(image_numpy.shape) == 3 and image_numpy.shape[0] in [1, 3]:  # 预期形状 (C, H, W)
            image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
        elif len(image_numpy.shape) == 2:  # 如果是灰度图像 (H, W)
            image_numpy = image_numpy * 255.0
        elif len(image_numpy.shape) == 3 and image_numpy.shape[2] in [1, 3]:  # 已经是 (H, W, C)
            image_numpy = image_numpy * 255.0
        else:
            raise ValueError(
                f"Unexpected shape for image_numpy: {image_numpy.shape}. Expected (C, H, W), (H, W), or (H, W, C).")
    else:
        if isinstance(image_tensor, torch.Tensor):
            image_numpy = image_tensor.float().numpy()
        else:
            image_numpy = image_tensor.astype(float)
        image_numpy = (np.transpose(image_numpy, (0, 2, 3, 1))) * 255.0

    image_numpy = np.clip(image_numpy, 0, 255)

    return image_numpy


def postprocess_bayer_v3(raw, img4c):
    out_srgb = process.raw2rgb_postprocess(img4c.detach(), raw)
    return out_srgb

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def metainfo(rawpath):
    with open(rawpath, 'rb') as f:
        tags = exifread.process_file(f)
        _, suffix = os.path.splitext(os.path.basename(rawpath))

        if suffix == '.dng':
            expo = eval(str(tags['Image ExposureTime']))
            iso = eval(str(tags['Image ISOSpeedRatings']))
        else:
            expo = eval(str(tags['EXIF ExposureTime']))
            iso = eval(str(tags['EXIF ISOSpeedRatings']))

        # print('ISO: {}, ExposureTime: {}'.format(iso, expo))
    return iso, expo

def metainfo_LRD(rawpath):
    with open(rawpath, 'rb') as f:
        tags = exifread.process_file(f)
        _, suffix = os.path.splitext(os.path.basename(rawpath))

        noise_profile = eval(str(tags['Image Tag 0xC761']))
        cshot, cread = float(noise_profile[0][0]), float(noise_profile[1][0])

        if suffix == '.dng':
            expo = eval(str(tags['Image ExposureTime']))
            iso = eval(str(tags['Image ISOSpeedRatings']))
            ev = eval(str(tags['Image ExposureBiasValue']))
        else:
            expo = eval(str(tags['EXIF ExposureTime']))
            iso = eval(str(tags['EXIF ISOSpeedRatings']))
            ev = eval(str(tags['EXIF ExposureBiasValue']))

        # print('ISO: {}, ExposureTime: {}'.format(iso, expo))
    return iso, expo

def compute_expo_ratio_LRD(target_fn, input_fn, ISO):
    ISO = int(''.join([c for c in ISO if c.isdigit()]))
    # EV = -float(EV)
    in_exposure = float(input_fn.split('_')[-1][:-5])
    gt_exposure = float(target_fn.split('_')[-1][:-5])
    # ratio = min(gt_exposure / in_exposure, 300)
    ratio = (100 * gt_exposure) / (ISO * in_exposure)
    return ratio

def crop_center(img, cropx, cropy):
    _, y, x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[:, starty:starty+cropy,startx:startx+cropx]


class SIDDataset(BaseDataset):
    def __init__(
        self, datadir, paired_fns, noise_maker, size=None, flag=None, augment=True, repeat=1, cfa='bayer', memorize=True,
        stage_in='raw', stage_out='raw', gt_wb=False, CRF=None, continuous_noise=False, patch_size=512):
        super(SIDDataset, self).__init__()
        assert cfa == 'bayer' or cfa == 'bayer'
        self.size = size
        self.noise_maker = noise_maker
        self.datadir = datadir
        self.paired_fns = paired_fns
        self.flag = flag
        self.augment = augment
        self.patch_size = patch_size
        self.repeat = repeat
        self.cfa = cfa
        self.continuous_noise = continuous_noise


        self.pack_raw = pack_raw_bayer if cfa == 'bayer' else pack_raw_bayer

        assert stage_in in ['raw', 'srgb']
        assert stage_out in ['raw', 'srgb']                
        self.stage_in = stage_in
        self.stage_out = stage_out
        self.gt_wb = gt_wb     
        self.CRF = CRF   

        if size is not None:
            self.paired_fns = self.paired_fns[:size]
        
        self.memorize = memorize
        self.target_dict = {}
        self.target_dict_aux = {}
        self.input_dict = {}

    def __getitem__(self, i):
        i = i % len(self.paired_fns)

        # target_fn, input_fn, ISO = self.paired_fns[i]
        input_fn, target_fn  = self.paired_fns[i]

        # input_path = join(self.datadir, 'Noisy', os.path.basename(input_fn))
        # target_path = join(self.datadir, 'Clean', os.path.basename(target_fn))
        input_path = join(self.datadir, 'short', input_fn)
        target_path = join(self.datadir, 'long', target_fn)

        # print("路径：",input_path)
        # print(target_path)

        # iso, expo = metainfo_LRD(input_path)
        iso, expo = metainfo(input_path)
        K = self.noise_maker.ISO_to_K(iso)

        # ratio = compute_expo_ratio_LRD(target_fn, input_fn, ISO)
        ratio = compute_expo_ratio(input_fn, target_fn)
        CRF = self.CRF

        if self.memorize:
            if target_fn not in self.target_dict:
                with rawpy.imread(target_path) as raw_target:                    
                    target_image = self.pack_raw(raw_target)    
                    wb, ccm = process.read_wb_ccm(raw_target)
                    if self.stage_out == 'srgb':
                        target_image = process.raw2rgb(target_image, raw_target, CRF)
                    self.target_dict[target_fn] = target_image
                    self.target_dict_aux[target_fn] = (wb, ccm)

            if input_fn not in self.input_dict:
                with rawpy.imread(input_path) as raw_input:
                    input_image = self.pack_raw(raw_input) * ratio
                    if self.stage_in == 'srgb':
                        if self.gt_wb:
                            wb, ccm = self.target_dict_aux[target_fn]
                            input_image = process.raw2rgb_v2(input_image, wb, ccm, CRF)
                        else:
                            input_image = process.raw2rgb(input_image, raw_input, CRF)
                    self.input_dict[input_fn] = input_image

            input_image = self.input_dict[input_fn]
            target_image = self.target_dict[target_fn]
            (wb, ccm) = self.target_dict_aux[target_fn]
        else:
            with rawpy.imread(target_path) as raw_target:                    
                target_image = self.pack_raw(raw_target)    
                wb, ccm = process.read_wb_ccm(raw_target)
                if self.stage_out == 'srgb':
                    target_image = process.raw2rgb(target_image, raw_target, CRF)

            with rawpy.imread(input_path) as raw_input:
                input_image = self.pack_raw(raw_input) * ratio
                if self.stage_in == 'srgb':
                    if self.gt_wb:
                        input_image = process.raw2rgb_v2(input_image, wb, ccm, CRF)
                    else:
                        input_image = process.raw2rgb(input_image, raw_input, CRF)  

        if self.augment:
            # print("????????????????????????????????")
            H = input_image.shape[1]
            W = target_image.shape[2]

            ps = self.patch_size

            xx = np.random.randint(0, W - ps)
            yy = np.random.randint(0, H - ps)

            input = input_image[:, yy:yy + ps, xx:xx + ps]
            target = target_image[:, yy:yy + ps, xx:xx + ps]

            if np.random.randint(2, size=1)[0] == 1:  # random flip
                input = np.flip(input, axis=1) # H
                target = np.flip(target, axis=1)
            if np.random.randint(2, size=1)[0] == 1:
                input = np.flip(input, axis=2) # W
                target = np.flip(target, axis=2)
            if np.random.randint(2, size=1)[0] == 1:  # random transpose
                input = np.transpose(input, (0, 2, 1))
                target = np.transpose(target, (0, 2, 1))
        else:
            input = input_image
            target = target_image

        # _, variance, _ = self.noise_maker(input, continuous=self.continuous_noise)
        # _, variance, _ = self.noise_maker(target, continuous=self.continuous_noise)

        input = np.maximum(np.minimum(input, 1.0), 0)
        target = np.maximum(np.minimum(target, 1.0), 0)
        input = np.ascontiguousarray(input)
        target = np.ascontiguousarray(target)

        # target_temp = np.maximum(np.minimum(target, 1.0), 0)
        # target_temp = np.ascontiguousarray(target_temp)

        # _, variance, _ = self.noise_maker(target, continuous=self.continuous_noise)

        variance = input

        # variance = input - target
        # variance = np.maximum(np.minimum(variance, 1.0), 0)
        # variance = np.ascontiguousarray(variance)

        # print(input)



        # if True:
        #     assert K > 0 and ratio > 0
        #     saturation_level = 16383 - 800
        #     target_ratio = ratio // 20
            
        #     input_photon = (input * saturation_level / ratio / K)
        #     increased_photon = np.random.poisson(target * saturation_level / ratio / K * (target_ratio-1))
        #     input = (increased_photon + input_photon) * K /saturation_level / target_ratio * ratio
        #     input = input.astype("float32")
        #     ratio = ratio // target_ratio
            
        dic =  {'input': input, 'target': target, 'variance': variance, 'fn': input_fn, 'cfa': self.cfa, 'rawpath': target_path, "ratio": ratio, "K":K}

        if self.flag is not None:
            dic.update(self.flag)
        
        return dic

    def __len__(self):
        return len(self.paired_fns) * self.repeat


def compute_expo_ratio(input_fn, target_fn):        
    in_exposure = float(input_fn.split('_')[-1][:-5])
    gt_exposure = float(target_fn.split('_')[-1][:-5])
    ratio = min(gt_exposure / in_exposure, 300)
    return ratio


def pack_raw_bayer(raw):
    #pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    raw_pattern = raw.raw_pattern
    R = np.where(raw_pattern==0)
    G1 = np.where(raw_pattern==1)
    B = np.where(raw_pattern==2)
    G2 = np.where(raw_pattern==3)
    
    white_point = 16383
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.stack((im[R[0][0]:H:2,R[1][0]:W:2], #RGBG
                    im[G1[0][0]:H:2,G1[1][0]:W:2],
                    im[B[0][0]:H:2,B[1][0]:W:2],
                    im[G2[0][0]:H:2,G2[1][0]:W:2]), axis=0).astype(np.float32)

    black_level = np.array(raw.black_level_per_channel)[:,None,None].astype(np.float32)

    out = (out - black_level) / (white_point - black_level)
    out = np.clip(out, 0, 1)
    
    return out

def pack_raw_LRD(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)

    white_point = 65535.0

    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.stack((im[0:H:2, 0:W:2],  # RGGB
                    im[0:H:2, 1:W:2],
                    im[1:H:2, 0:W:2],
                    im[1:H:2, 1:W:2]), axis=0).astype(np.float32)
    black_level = np.array(raw.black_level_per_channel)[:, None, None].astype(np.float32)
    out = (out - black_level) / (white_point - black_level)
    out = np.clip(out, 0, 1)
    return out


def pack_raw_xtrans(raw):
    # pack X-Trans image to 9 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = (im - 1024) / (16383 - 1024)  # subtract the black level
    im = np.clip(im, 0, 1)

    img_shape = im.shape
    H = (img_shape[0] // 6) * 6
    W = (img_shape[1] // 6) * 6

    out = np.zeros((9, H // 3, W // 3), dtype=np.float32)

    # 0 R
    out[0, 0::2, 0::2] = im[0:H:6, 0:W:6]
    out[0, 0::2, 1::2] = im[0:H:6, 4:W:6]
    out[0, 1::2, 0::2] = im[3:H:6, 1:W:6]
    out[0, 1::2, 1::2] = im[3:H:6, 3:W:6]

    # 1 G
    out[1, 0::2, 0::2] = im[0:H:6, 2:W:6]
    out[1, 0::2, 1::2] = im[0:H:6, 5:W:6]
    out[1, 1::2, 0::2] = im[3:H:6, 2:W:6]
    out[1, 1::2, 1::2] = im[3:H:6, 5:W:6]

    # 1 B
    out[2, 0::2, 0::2] = im[0:H:6, 1:W:6]
    out[2, 0::2, 1::2] = im[0:H:6, 3:W:6]
    out[2, 1::2, 0::2] = im[3:H:6, 0:W:6]
    out[2, 1::2, 1::2] = im[3:H:6, 4:W:6]

    # 4 R
    out[3, 0::2, 0::2] = im[1:H:6, 2:W:6]
    out[3, 0::2, 1::2] = im[2:H:6, 5:W:6]
    out[3, 1::2, 0::2] = im[5:H:6, 2:W:6]
    out[3, 1::2, 1::2] = im[4:H:6, 5:W:6]

    # 5 B
    out[4, 0::2, 0::2] = im[2:H:6, 2:W:6]
    out[4, 0::2, 1::2] = im[1:H:6, 5:W:6]
    out[4, 1::2, 0::2] = im[4:H:6, 2:W:6]
    out[4, 1::2, 1::2] = im[5:H:6, 5:W:6]

    out[5, :, :] = im[1:H:3, 0:W:3]
    out[6, :, :] = im[1:H:3, 1:W:3]
    out[7, :, :] = im[2:H:3, 0:W:3]
    out[8, :, :] = im[2:H:3, 1:W:3]
    return out


class SynDataset(BaseDataset):  # generate noisy image only 
    def __init__(self, dataset, size=None, flag=None, noise_maker=None, repeat=1, cfa='bayer', num_burst=1, continuous_noise=False):
        super(SynDataset, self).__init__()        
        self.size = size
        self.dataset = dataset
        self.flag = flag
        self.repeat = repeat
        self.noise_maker = noise_maker
        self.cfa = cfa
        self.num_burst = num_burst
        self.continuous_noise = continuous_noise
        
    def __getitem__(self, i):
        if self.size is not None:
            i = i % self.size
        else:
            i = i % len(self.dataset)
            
        data, metadata = self.dataset[i]

        if self.num_burst > 1:            
            inputs = []
            params = self.noise_maker._sample_params()     
            for k in range(self.num_burst):           
                # inputs.append(self.noise_maker(data))
                inputs.append(self.noise_maker(data, params=params, continuous=self.continuous_noise))
            input = np.concatenate(inputs, axis=0)
        else:
            input, variance, params = self.noise_maker(data, continuous=self.continuous_noise)

        # variance = np.maximum(np.minimum(input, 1,0), 0)


        input = np.maximum(np.minimum(input, 1.0), 0)
        input = np.ascontiguousarray(input)
        variance = np.ascontiguousarray(variance)

        return input, variance, params
        
    def __len__(self):
        size = self.size or len(self.dataset)
        return int(size * self.repeat)
    

class ISPDataset(BaseDataset):
    def __init__(self, dataset, noise_maker=None, cfa='bayer', meta_info=None, CRF=None):
        super(ISPDataset, self).__init__()        
        self.dataset = dataset
        self.noise_maker = noise_maker
        self.cfa = cfa

        if meta_info is None:
            self.meta_info = dataset.meta
        else:
            self.meta_info = meta_info

        self.CRF = CRF
        
    def __getitem__(self, i):
        data = self.dataset[i]
        (wb, ccm) = self.meta_info[i]
        # (wb, ccm, *rest) = self.meta_info[i]
        CRF = self.CRF
        # print("///////////////////////////////////////////////")
        # print(f"Type of y: {type(data)}, Value of y: {data}")

        if self.noise_maker is not None:
            input = self.noise_maker(data)
        else:
            input = data

        input = np.maximum(np.minimum(input, 1.0), 0)
        input = process.raw2rgb_v2(input, wb, ccm, CRF)
        input = np.maximum(np.minimum(input, 1.0), 0)
        input = np.ascontiguousarray(input)

        return input

    def __len__(self):
        return len(self.dataset)    


class ELDTrainDataset(BaseDataset):
    def __init__(self, target_dataset, input_datasets, size=None, flag=None, augment=True, cfa='bayer', syn_noise=False, CRF=None):
        super(ELDTrainDataset, self).__init__()
        self.size = size
        self.target_dataset = target_dataset
        self.input_datasets = input_datasets
        self.flag = flag
        self.augment = augment
        self.cfa = cfa
        self.syn_noise = syn_noise # synthetic possion noise
        self.CRF = CRF

    def __getitem__(self, i):
        N = len(self.input_datasets)
        # input_image, variance, noise_params = self.input_datasets[i%N][i//N]
        input_image, variance,noise_params = self.input_datasets[i%N][i//N]
        target_image, _ = self.target_dataset[i//N]

        target = target_image 
        input = input_image

        # input = np.maximum(np.minimum(input, 1.0), 0)
        # input = np.ascontiguousarray(input)

        # 扩展到 0-255
        # normalized_image = (variance * 255).astype(np.uint8)
        # normalized_image = (input * 255).astype(np.uint8)

        # variance = tensor2im(normalized_image)
        # variance = tensor2im(target)
        #
        # plt.imshow(variance)
        # plt.title('Image Visualization with Matplotlib')
        # plt.axis('off')  # 隐藏坐标轴
        # plt.show()

        # # 使用 matplotlib 显示归一化的图像
        # plt.imshow(variance, cmap='gray')
        # plt.title('Normalized RAW Image')
        # plt.axis('off')
        # plt.show()

        # 保存生成的残差图到文件
        # plt.imsave('residual_image5.png', variance.astype('uint8'))

        if self.augment:
            W = target_image.shape[2]
            H = target_image.shape[1]

            if np.random.randint(2, size=1)[0] == 1:  # random flip
                target = np.flip(target, axis=1)
                input = np.flip(input, axis=1)
                # variance = np.flip(variance, axis=1)
            if np.random.randint(2, size=1)[0] == 1:
                target = np.flip(target, axis=2)
                input = np.flip(input, axis=2)
                # variance = np.flip(variance, axis=2)
            if np.random.randint(2, size=1)[0] == 1:  # random transpose
                target = np.transpose(target, (0, 2, 1))
                input = np.transpose(input, (0, 2, 1))
                # variance = np.transpose(variance, (0, 2, 1))

        input = np.maximum(np.minimum(input, 1.0), 0)
        input = np.ascontiguousarray(input)
        target = np.ascontiguousarray(target)

        # input_img = torch.from_numpy(input)
        # target_img_numpy = target
        # target_img = torch.from_numpy(target)
        # variance_img = torch.from_numpy(variance)
        # input_img = input_img.unsqueeze(0)
        # target_img = target_img.unsqueeze(0)
        # variance_img = variance_img.unsqueeze(0)
        # # variance = np.maximum(np.minimum(variance, 1.0), 0)
        # # variance = np.ascontiguousarray(variance)
        # input_img = tensor2im(postprocess_bayer_v3(target_img_numpy, input_img))
        # target_img = tensor2im(postprocess_bayer_v3(target_img_numpy,target_img))
        # variance_img = tensor2im(postprocess_bayer_v3(target_img_numpy,variance_img))
        variance = input
        
        ratio = noise_params["ratio"]
        K = noise_params["K"]

        if self.syn_noise:
            assert K > 0 and ratio > 0
            saturation_level = 16383 - 800
            target_ratio = 1 / random.uniform(1/ratio, 1/100)
            
            input_photon = (input * saturation_level / ratio / K)
            increased_photon = np.random.poisson(target * saturation_level / ratio / K * (ratio/target_ratio-1))
            input = (increased_photon + input_photon) * K /saturation_level * target_ratio
            input = input.astype("float32")
        
        dic =  {'input': input, 'target': target, 'variance':variance, "ratio": ratio, "K": K}
        if self.flag is not None:
            dic.update(self.flag)
        
        return dic

    def __len__(self):
        size = self.size or len(self.target_dataset) * len(self.input_datasets)
        return size

class ELDEvalDataset(BaseDataset):
    def __init__(self, basedir, camera_suffix, noiser_maker, scenes=None, img_ids=None):
        super(ELDEvalDataset, self).__init__()
        self.basedir = basedir
        self.camera_suffix = camera_suffix # ('Canon', '.CR2')
        self.scenes = scenes
        self.img_ids = img_ids
        # self.input_dict = {}
        # self.target_dict = {}
        self.noise_maker = noiser_maker
        
    def __getitem__(self, i):
        camera, suffix = self.camera_suffix
        
        scene_id = i // len(self.img_ids)
        img_id = i % len(self.img_ids)

        scene = 'scene-{}'.format(self.scenes[scene_id])

        datadir = join(self.basedir, camera, scene)

        input_path = join(datadir, 'IMG_{:04d}{}'.format(self.img_ids[img_id], suffix))

        gt_ids = np.array([1, 6, 11, 16])
        ind = np.argmin(np.abs(self.img_ids[img_id] - gt_ids))
        
        target_path = join(datadir, 'IMG_{:04d}{}'.format(gt_ids[ind], suffix))

        iso, expo = metainfo(target_path)
        target_expo = iso * expo
        iso, expo = metainfo(input_path)

        ratio = target_expo / (iso * expo)
        K = self.noise_maker.ISO_to_K(iso)
        with rawpy.imread(input_path) as raw:
            input = pack_raw_bayer(raw) * ratio            

        with rawpy.imread(target_path) as raw:
            target = pack_raw_bayer(raw)

        input = np.maximum(np.minimum(input, 1.0), 0)
        target = np.maximum(np.minimum(target, 1.0), 0)
        input = np.ascontiguousarray(input)
        target = np.ascontiguousarray(target)

        # target_temp = target

        variance = input
        # variance = np.maximum(np.min=imum(variance, 1.0), 0)
        # variance = np.ascontiguousarray(variance)

        data = {'input': input, 'target': target, 'variance':variance, 'fn':input_path, 'rawpath': target_path, 'ratio': ratio, "K": K}
        
        return data

    def __len__(self):
        return len(self.scenes) * len(self.img_ids)

