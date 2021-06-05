import os
import subprocess
import torch.utils.data as data
import numpy as np
import time
import torch
from natsort import natsort
import imageio
import pickle
import glob
import sys
sys.path.append('../NTIRE21_Learning_SR_Space/')
from imresize import imresize


def fiFindByWildcard(wildcard):
    return natsort.natsorted(glob.glob(wildcard, recursive=True))

class LRHR_IMGDataset(data.Dataset):
    def __init__(self, opt):
        super(LRHR_IMGDataset, self).__init__()
        self.opt = opt
        self.crop_size = opt.get("GT_size", None)
        self.scale = opt['scale']
        self.random_scale_list = [1]

        hr_file_path = opt["dataroot_GT"]
        y_labels_file_path = opt['dataroot_y_labels']
        
        gpu = True
        augment = True
        
        self.use_flip = opt["use_flip"] if "use_flip" in opt.keys() else False
        self.use_rot = opt["use_rot"] if "use_rot" in opt.keys() else False
        self.use_crop = opt["use_crop"] if "use_crop" in opt.keys() else False
        self.center_crop_hr_size = opt.get("center_crop_hr_size", None)

        n_max = opt["n_max"] if "n_max" in opt.keys() else int(1e8)

        t = time.time()
        
        self.hr_paths = fiFindByWildcard(os.path.join(hr_file_path, '*.png'))

        t = time.time() - t
        
        self.gpu = gpu
        self.augment = augment

        self.measures = None
    
    def imread(self, img_path):
        img = imageio.imread(img_path)
        if len(img.shape) == 2:
            img = np.stack([img, ] * 3, axis=2)
        return img

    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, item):
        hr = self.imread(self.hr_paths[item])
        hr = random_crop(hr, self.crop_size)
        lr = imresize(hr, scalar_scale = 1 / self.scale)
        
        hr = np.transpose(hr, [2, 0, 1])
        lr = np.transpose(lr, [2, 0, 1])

        if self.center_crop_hr_size:
            hr, lr = center_crop(hr, self.center_crop_hr_size), center_crop(lr, self.center_crop_hr_size // self.scale)

        if self.use_flip:
            hr, lr = random_flip(hr, lr)

        if self.use_rot:
            hr, lr = random_rotation(hr, lr)

        hr = hr / 255.0
        lr = lr / 255.0

        if self.measures is None or np.random.random() < 0.05:
            if self.measures is None:
                self.measures = {}
            self.measures['hr_means'] = np.mean(hr)
            self.measures['hr_stds'] = np.std(hr)
            self.measures['lr_means'] = np.mean(lr)
            self.measures['lr_stds'] = np.std(lr)

        hr = torch.Tensor(hr)
        lr = torch.Tensor(lr)

        # if self.gpu:
        #    hr = hr.cuda()
        #    lr = lr.cuda()

        return {'LQ': lr, 'GT': hr, 'LQ_path': str(item), 'GT_path': str(item)}

    def print_and_reset(self, tag):
        m = self.measures
        kvs = []
        for k in sorted(m.keys()):
            kvs.append("{}={:.2f}".format(k, m[k]))
        print("[KPI] " + tag + ": " + ", ".join(kvs))
        self.measures = None


def random_flip(img, seg):
    random_choice = np.random.choice([True, False])
    img = img if random_choice else np.flip(img, 2).copy()
    seg = seg if random_choice else np.flip(seg, 2).copy()
    return img, seg


def random_rotation(img, seg):
    random_choice = np.random.choice([0, 1, 3])
    img = np.rot90(img, random_choice, axes=(1, 2)).copy()
    seg = np.rot90(seg, random_choice, axes=(1, 2)).copy()
    return img, seg

def random_crop(img, size):
    h, w, c = img.shape

    h_start = np.random.randint(0, h - size)
    h_end = h_start + size

    w_start = np.random.randint(0, w - size)
    w_end = w_start + size

    return img[h_start:h_end, w_start:w_end]


def center_crop(img, size):
    assert img.shape[1] == img.shape[2], img.shape
    border_double = img.shape[1] - size
    assert border_double % 2 == 0, (img.shape, size)
    border = border_double // 2
    return img[:, border:-border, border:-border]


def center_crop_tensor(img, size):
    assert img.shape[2] == img.shape[3], img.shape
    border_double = img.shape[2] - size
    assert border_double % 2 == 0, (img.shape, size)
    border = border_double // 2
    return img[:, :, border:-border, border:-border]
