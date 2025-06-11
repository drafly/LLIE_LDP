import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def read_fns(filename):
    with open(filename) as f:
        fns = f.readlines()
        fns = [fn.strip() for fn in fns]
    return fns


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def read_paired_fns(filename):
    with open(filename) as f:
        fns = f.readlines()
        fns = [tuple(fn.strip().split(' ')) for fn in fns]
    return fns


def compute_expo_ratio_LRD(input_fn, target_fn, ISO, EV):
    ISO = int(''.join([c for c in ISO if c.isdigit()]))
    EV = -int(''.join([c for c in EV if c.isdigit()]))
    in_exposure = float(input_fn.split('_')[-1][:-5])
    gt_exposure = float(target_fn.split('_')[-1][:-5])
    # ratio = min(gt_exposure / in_exposure, 300)
    ratio = (100 * gt_exposure) / (ISO * in_exposure)
    return EV