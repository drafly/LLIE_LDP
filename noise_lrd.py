import numpy as np
import scipy.stats as stats
from os.path import join
from scipy.stats import tukeylambda
import torch
import torch.distributions as tdist


def log(string, log=None, str=False, end='\n', notime=False):
    log_string = f'{time.strftime("%Y-%m-%d %H:%M:%S")} >>  {string}' if not notime else string
    print(log_string)
    if log is not None:
        with open(log, 'a+') as f:
            f.write(log_string + '\n')
    else:
        pass

    if str:
        return string + end

class RawPacker:
    def __init__(self, cfa='lrd'):
        self.cfa = cfa

    def pack_raw_bayer(self, cfa_img):
        # pack Bayer image to 4 channels
        img_shape = cfa_img.shape
        H = img_shape[0]
        W = img_shape[1]

        out = np.stack((cfa_img[0:H:2, 0:W:2],  # RGBG
                        cfa_img[0:H:2, 1:W:2],
                        cfa_img[1:H:2, 1:W:2],
                        cfa_img[1:H:2, 0:W:2]), axis=0).astype(np.float32)
        return out

    def pack_raw_xtrans(self, cfa_img):
        # pack X-Trans image to 9 channels
        img_shape = cfa_img.shape
        H = (img_shape[0] // 6) * 6
        W = (img_shape[1] // 6) * 6

        out = np.zeros((9, H // 3, W // 3), dtype=np.float32)

        # 0 R
        out[0, 0::2, 0::2] = cfa_img[0:H:6, 0:W:6]
        out[0, 0::2, 1::2] = cfa_img[0:H:6, 4:W:6]
        out[0, 1::2, 0::2] = cfa_img[3:H:6, 1:W:6]
        out[0, 1::2, 1::2] = cfa_img[3:H:6, 3:W:6]

        # 1 G
        out[1, 0::2, 0::2] = cfa_img[0:H:6, 2:W:6]
        out[1, 0::2, 1::2] = cfa_img[0:H:6, 5:W:6]
        out[1, 1::2, 0::2] = cfa_img[3:H:6, 2:W:6]
        out[1, 1::2, 1::2] = cfa_img[3:H:6, 5:W:6]

        # 1 B
        out[2, 0::2, 0::2] = cfa_img[0:H:6, 1:W:6]
        out[2, 0::2, 1::2] = cfa_img[0:H:6, 3:W:6]
        out[2, 1::2, 0::2] = cfa_img[3:H:6, 0:W:6]
        out[2, 1::2, 1::2] = cfa_img[3:H:6, 4:W:6]

        # 4 R
        out[3, 0::2, 0::2] = cfa_img[1:H:6, 2:W:6]
        out[3, 0::2, 1::2] = cfa_img[2:H:6, 5:W:6]
        out[3, 1::2, 0::2] = cfa_img[5:H:6, 2:W:6]
        out[3, 1::2, 1::2] = cfa_img[4:H:6, 5:W:6]

        # 5 B
        out[4, 0::2, 0::2] = cfa_img[2:H:6, 2:W:6]
        out[4, 0::2, 1::2] = cfa_img[1:H:6, 5:W:6]
        out[4, 1::2, 0::2] = cfa_img[4:H:6, 2:W:6]
        out[4, 1::2, 1::2] = cfa_img[5:H:6, 5:W:6]

        out[5, :, :] = cfa_img[1:H:3, 0:W:3]
        out[6, :, :] = cfa_img[1:H:3, 1:W:3]
        out[7, :, :] = cfa_img[2:H:3, 0:W:3]
        out[8, :, :] = cfa_img[2:H:3, 1:W:3]
        return out

    def pack_raw_DJI(raw):
        # pack Bayer image to 4 channels
        im = raw.raw_image_visible.astype(np.float32)
        raw_pattern = raw.raw_pattern
        R = np.where(raw_pattern == 0)
        G1 = np.where(raw_pattern == 1)
        B = np.where(raw_pattern == 2)
        G2 = np.where(raw_pattern == 3)

        white_point = 65535.0

        img_shape = im.shape
        H = img_shape[0]
        W = img_shape[1]

        out = np.stack((im[0:H:2, 0:W:2],  # RGGB
                        im[0:H:2, 1:W:2],
                        im[1:H:2, 0:W:2],
                        im[1:H:2, 1:W:2]), axis=0).astype(np.float32)

        # out = np.stack((im[R[0][0]:H:2, R[1][0]:W:2],  # RGGB
        #                 im[G1[0][0]:H:2, G1[1][0]:W:2],
        #                 im[G2[0][0]:H:2, G2[1][0]:W:2],
        #                 im[B[0][0]:H:2, B[1][0]:W:2]), axis=0).astype(np.float32)

        black_level = np.array(raw.black_level_per_channel)[:, None, None].astype(np.float32)

        # if max(raw.black_level_per_channel) != min(raw.black_level_per_channel):
        #     black_level = 2**round(np.log2(np.max(black_level)))
        # print(black_level)

        out = (out - black_level) / (white_point - black_level)
        out = np.clip(out, 0, 1)
        return out

    def unpack_raw_bayer(self, img):
        # unpack 4 channels to Bayer image
        img4c = img
        _, h, w = img.shape

        H = int(h * 2)
        W = int(w * 2)

        cfa_img = np.zeros((H, W), dtype=np.float32)

        cfa_img[0:H:2, 0:W:2] = img4c[0, :, :]
        cfa_img[0:H:2, 1:W:2] = img4c[1, :, :]
        cfa_img[1:H:2, 1:W:2] = img4c[2, :, :]
        cfa_img[1:H:2, 0:W:2] = img4c[3, :, :]

        return cfa_img

    def unpack_raw_xtrans(self, img):
        img9c = img
        _, h, w = img.shape

        H = int(h * 3)
        W = int(w * 3)

        cfa_img = np.zeros((H, W), dtype=np.float32)

        # 0 R
        cfa_img[0:H:6, 0:W:6] = img9c[0, 0::2, 0::2]
        cfa_img[0:H:6, 4:W:6] = img9c[0, 0::2, 1::2]
        cfa_img[3:H:6, 1:W:6] = img9c[0, 1::2, 0::2]
        cfa_img[3:H:6, 3:W:6] = img9c[0, 1::2, 1::2]

        # 1 G
        cfa_img[0:H:6, 2:W:6] = img9c[1, 0::2, 0::2]
        cfa_img[0:H:6, 5:W:6] = img9c[1, 0::2, 1::2]
        cfa_img[3:H:6, 2:W:6] = img9c[1, 1::2, 0::2]
        cfa_img[3:H:6, 5:W:6] = img9c[1, 1::2, 1::2]

        # 1 B
        cfa_img[0:H:6, 1:W:6] = img9c[2, 0::2, 0::2]
        cfa_img[0:H:6, 3:W:6] = img9c[2, 0::2, 1::2]
        cfa_img[3:H:6, 0:W:6] = img9c[2, 1::2, 0::2]
        cfa_img[3:H:6, 4:W:6] = img9c[2, 1::2, 1::2]

        # 4 R
        cfa_img[1:H:6, 2:W:6] = img9c[3, 0::2, 0::2]
        cfa_img[2:H:6, 5:W:6] = img9c[3, 0::2, 1::2]
        cfa_img[5:H:6, 2:W:6] = img9c[3, 1::2, 0::2]
        cfa_img[4:H:6, 5:W:6] = img9c[3, 1::2, 1::2]

        # 5 B
        cfa_img[2:H:6, 2:W:6] = img9c[4, 0::2, 0::2]
        cfa_img[1:H:6, 5:W:6] = img9c[4, 0::2, 1::2]
        cfa_img[4:H:6, 2:W:6] = img9c[4, 1::2, 0::2]
        cfa_img[5:H:6, 5:W:6] = img9c[4, 1::2, 1::2]

        cfa_img[1:H:3, 0:W:3] = img9c[5, :, :]
        cfa_img[1:H:3, 1:W:3] = img9c[6, :, :]
        cfa_img[2:H:3, 0:W:3] = img9c[7, :, :]
        cfa_img[2:H:3, 1:W:3] = img9c[8, :, :]

        return cfa_img

    def pack_raw(self, cfa_img):
        if self.cfa == 'bayer':
            out = self.pack_raw_bayer(cfa_img)
        elif self.cfa == 'xtrans':
            out = self.pack_raw_xtrans(cfa_img)
        elif self.cfa == 'lrd':
            out = self.pack_raw_DJI(cfa_img)
        else:
            raise NotImplementedError
        return out

    def unpack_raw(self, img):
        if self.cfa == 'bayer':
            out = self.unpack_raw_bayer(img)
        elif self.cfa == 'xtrans':
            out = self.unpack_raw_xtrans(img)
        else:
            raise NotImplementedError
        return out
def get_camera_noisy_params(camera_type=None):
    cam_noisy_params = {'NikonD850': {
        'Kmin': 1.2, 'Kmax': 2.4828, 'lam': -0.26, 'q': 1 / (2 ** 14), 'wp': 16383, 'bl': 512,
        'sigTLk': 0.906, 'sigTLb': -0.6754, 'sigTLsig': 0.035165,
        'sigRk': 0.8322, 'sigRb': -2.3326, 'sigRsig': 0.301333,
        'sigGsk': 0.8322, 'sigGsb': -0.1754, 'sigGssig': 0.035165,
    }, 'IMX686': {  # ISO-640~6400
        'Kmin': -0.19118, 'Kmax': 2.16820, 'lam': 0.102, 'q': 1 / (2 ** 10), 'wp': 1023, 'bl': 64,
        'sigTLk': 0.85187, 'sigTLb': 0.07991, 'sigTLsig': 0.02921,
        'sigRk': 0.87611, 'sigRb': -2.11455, 'sigRsig': 0.03274,
        'sigGsk': 0.85187, 'sigGsb': 0.67991, 'sigGssig': 0.02921,
    }, 'SonyA7S2_lowISO': {
        'Kmin': -1.67214, 'Kmax': 0.42228, 'lam': -0.026, 'q': 1 / (2 ** 14), 'wp': 16383, 'bl': 512,
        'sigRk': 0.78782, 'sigRb': -0.34227, 'sigRsig': 0.02832,
        'sigTLk': 0.74043, 'sigTLb': 0.86182, 'sigTLsig': 0.00712,
        'sigGsk': 0.82966, 'sigGsb': 1.49343, 'sigGssig': 0.00359,
        'sigReadk': 0.82879, 'sigReadb': 1.50601, 'sigReadsig': 0.00362,
        'uReadk': 0.01472, 'uReadb': 0.01129, 'uReadsig': 0.00034,
    }, 'SonyA7S2_highISO': {
        'Kmin': 0.64567, 'Kmax': 2.51606, 'lam': -0.025, 'q': 1 / (2 ** 14), 'wp': 16383, 'bl': 512,
        'sigRk': 0.62945, 'sigRb': -1.51040, 'sigRsig': 0.02609,
        'sigTLk': 0.74901, 'sigTLb': -0.12348, 'sigTLsig': 0.00638,
        'sigGsk': 0.82878, 'sigGsb': 0.44162, 'sigGssig': 0.00153,
        'sigReadk': 0.82645, 'sigReadb': 0.45061, 'sigReadsig': 0.00156,
        'uReadk': 0.00385, 'uReadb': 0.00674, 'uReadsig': 0.00039,
    }, 'CRVD': {
        'Kmin': 1.31339, 'Kmax': 3.95448, 'lam': 0.015, 'q': 1 / (2 ** 12), 'wp': 4095, 'bl': 240,
        'sigRk': 0.93368, 'sigRb': -2.19692, 'sigRsig': 0.02473,
        'sigGsk': 0.95387, 'sigGsb': 0.01552, 'sigGssig': 0.00855,
        'sigTLk': 0.95495, 'sigTLb': 0.01618, 'sigTLsig': 0.00790
    }}
    if camera_type in cam_noisy_params:
        return cam_noisy_params[camera_type]
    else:
        log(f'''Warning: we have not test the noisy parameters of camera "{camera_type}". Now we use NikonD850's parameters to test.''')
        return cam_noisy_params['NikonD850']

class NoiseModelBase:  # base class
    def __call__(self, y, params=None, continuous=False):
        if params is None:
            K, sigTL, sigR, sigGs, bias, lam, q, ratio, wp, bl = self._sample_params()
        else:
            K, sigTL, sigR, sigGs, bias, lam, q, ratio, wp, bl = params

        noise_levels = {}
        MultiFrameMean = 1

        y = y * (wp - bl)

        y = y / ratio

        MFM = MultiFrameMean ** 0.5

        if "u" in self.model:  # quantization noise
            y = y + (np.random.uniform(0, 1, y.shape) - 0.5)
            y = y.clip(0)
            quantization_noise_variance = 1 / 12
            noise_levels['quantization'] = quantization_noise_variance
        if 'P' in self.model:

            noisy_shot = np.random.poisson(MFM * y / K).astype(np.float32) * K / MFM


        elif 'p' in self.model:
            z = y + np.random.randn(*y.shape).astype(np.float32) * np.sqrt(np.maximum(K * y, 1e-10))
            poisson_noise_variance_p = np.mean(K * y)
            noise_levels['poisson_p'] = poisson_noise_variance_p
        else:
            z = y

        if 'r' in self.model:  # row noise
            z = self.raw_packer.unpack_raw(z)

        if 'g' in self.model:
            noisy_read = stats.norm.rvs(scale=sigGs / MFM, size=y.shape).astype(np.float32)

        else:
            z = z
        z = (noisy_shot + noisy_read) / (wp - bl)
        z = np.clip(z, 0, 1)
        z = z * ratio

        return z.astype(np.float32),z.astype(np.float32), {"K": K, "ratio": ratio}


# Only support baseline noise models: G / G+P / G+P*
def sample_params_max(camera_type, ratio, iso):
    pass


class NoiseModel(NoiseModelBase):
    def __init__(self, model='g', cameras=None, include=None, exclude=None, cfa='lrd'):
        super().__init__()
        assert cfa in ['bayer', 'xtrans', 'lrd']
        assert include is None or exclude is None
        self.cameras = cameras or ['CanonEOS5D4', 'CanonEOS70D', 'CanonEOS700D', 'NikonD850', 'SonyA7S2']

        self.camera_params = {}

        self.model = model
        self.raw_packer = RawPacker(cfa)

    def ISO_to_K(self, ISO):
        camera_params = self.camera_params[self.cameras[0]]
        Kmin = camera_params['Kmin']
        Kmax = camera_params['Kmax']
        k = (ISO - 100) / (6400 - 100) * (Kmax - Kmin) + Kmin
        return k

    def ISO_to_K_LRD(self, ISO):
        k = np.log(ISO)
        return k

    def _sample_params(self):
        params = None
        params = get_camera_noisy_params(camera_type='IMX686')

        bias = 0
        log_K = params['Kmax'] + np.random.uniform(low=-0.01, high=+0.01)  # 增加一些扰动，以防测的不准
        K = np.exp(log_K)
        mu_TL = params['sigTLk'] * log_K + params['sigTLb']
        mu_R = params['sigRk'] * log_K + params['sigRb']
        mu_Gs = params['sigGsk'] * log_K + params['sigGsb'] if 'sigGsk' in params else 2 ** (-14)

        sigTL = np.exp(mu_TL)
        sigR = np.exp(mu_R)
        sigGs = np.exp(np.random.normal(loc=mu_Gs, scale=params['sigGssig']) if 'sigGssig' in params else mu_Gs)

        wp = params['wp']
        bl = params['bl']
        lam = params['lam']
        q = params['q']

        log_ratio = np.random.uniform(low=0, high=2.08)
        ratio = np.exp(log_ratio)

        return (K, sigTL, sigR,  sigGs,  bias, lam,  q, ratio, wp,  bl)