import nibabel as nib
from torch.utils.data import Dataset

from utils import *


class DatasetBrainSRT(Dataset):
    '''
    Get L/H for SISR.
    must paths_H and path_L both are provided.
    '''

    def __init__(self, opt):
        super(DatasetBrainSRT, self).__init__()
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.data_root = opt['dataroot_H']
        self.scale = opt['scale'] if opt['scale'] else 4
        self.paths = get_image_paths(self.data_root)

    def __getitem__(self, index):

        volumepath = self.paths[index]
        volume = nib.load(volumepath)
        volumeIn = np.array([volume.get_fdata()])
        volumeIn_t1 = volumeIn[:, :, :, :, 0].squeeze()
        volumeIn_t2 = volumeIn[:, :, :, :, 2].squeeze()

        max_d, min_d = max(volumeIn_t1.reshape(-1).max(), volumeIn_t2.reshape(-1).max()), min(
            volumeIn_t1.reshape(-1).min(), volumeIn_t2.reshape(-1).min())
        if (max_d - min_d) != 0:
            volumeIn_t1 = (volumeIn_t1 - volumeIn_t1.reshape(-1).min()) / (max_d - min_d)
            volumeIn_t2 = (volumeIn_t2 - volumeIn_t2.reshape(-1).min()) / (max_d - min_d)

        H, W, C = volumeIn_t2.shape
        volumeDown_t2 = self.degrade(volumeIn_t2).transpose(2, 1, 0)
        volumeDown_t1 = self.degrade(volumeIn_t1).transpose(2, 1, 0)

        idx = random.randint(0, C - 1)
        if self.opt['phase'] == 'train':
            volumeIn_t1 = volumeIn_t1[:, :, idx]
            volumeIn_t2 = volumeIn_t2[:, :, idx]
            volumeDown_t2 = volumeDown_t2[:, :, idx]
            volumeDown_t1 = volumeDown_t1[:, :, idx]

        volumeDown_t2 = single2tensor3(volumeDown_t2)
        volumeDown_t1 = single2tensor3(volumeDown_t1)
        volumeIn_t2 = single2tensor3(volumeIn_t2)
        volumeIn_t1 = single2tensor3(volumeIn_t1)
        # L:[I_in,Rc] H:[HR,HR]
        return {'L': [volumeDown_t2, volumeIn_t1],
                'H': [volumeIn_t2, get_gradient(volumeIn_t2.unsqueeze(1)).squeeze(1)], 'path': self.paths[index]}

    def __len__(self):
        return len(self.paths)

    def degrade(self, hr_data):
        norm_d = max(hr_data.reshape(-1)) - min(hr_data.reshape(-1))
        if self.scale == 4:
            imgfft = np.fft.fft2(hr_data.transpose(2, 1, 0))
            imgfft = np.fft.fftshift(imgfft)
            imgfft = imgfft[:, 90: 150, 90: 150]
            imgfft = np.fft.ifftshift(imgfft)
            imgifft = np.fft.ifft2(imgfft)
            img_out = abs(imgifft)

        if self.scale == 3:
            imgfft = np.fft.fft2(hr_data.transpose(2, 1, 0))
            imgfft = np.fft.fftshift(imgfft)
            imgfft = imgfft[:, 80: 160, 80: 160]
            imgfft = np.fft.ifftshift(imgfft)
            imgifft = np.fft.ifft2(imgfft)
            img_out = abs(imgifft)

        if self.scale == 2:
            imgfft = np.fft.fft2(hr_data.transpose(2, 1, 0))
            imgfft = np.fft.fftshift(imgfft)
            imgfft = imgfft[:, 60: 180, 60: 180]
            imgfft = np.fft.ifftshift(imgfft)
            imgifft = np.fft.ifft2(imgfft)
            img_out = abs(imgifft)
        if img_out.max() - img_out.min() != 0:
            img_out = (img_out - min(img_out.reshape(-1))) / (max(img_out.reshape(-1)) - min(img_out.reshape(-1)))
        return img_out


class DatasetIXIT(Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.data_root = opt['dataroot_H']
        self.HR_paths = get_image_paths(self.data_root)
        self.scale = opt['scale'] if opt['scale'] else 4
        self.img_size = opt['img_size']

    def __getitem__(self, index):
        volumepath_t1 = self.HR_paths[index]
        volumepath_t2 = volumepath_t1.replace('t1', 't2').replace('T1', 'T2')
        volume_t1 = nib.load(volumepath_t1)
        volume_t2 = nib.load(volumepath_t2)

        volumeIn_t1 = np.array([volume_t1.get_fdata()]).squeeze()
        volumeIn_t2 = np.array([volume_t2.get_fdata()]).squeeze()
        # 256,256,130 -> 240,240,130
        volumeIn_t1 = cv2.resize(volumeIn_t1, (self.img_size * self.scale, self.img_size * self.scale))
        # 256,256,130*2 -> 240,240,260
        volumeIn_t2 = volumeIn_t2.squeeze()
        volumeIn_t2 = cv2.resize(volumeIn_t2, (self.img_size * self.scale, self.img_size * self.scale))

        H, W, C = volumeIn_t2.shape

        idx = random.randint(0, C - 1)

        max_d, min_d = max(volumeIn_t1.reshape(-1).max(), volumeIn_t2.reshape(-1).max()), min(
            volumeIn_t1.reshape(-1).min(), volumeIn_t2.reshape(-1).min())
        if (max_d - min_d) != 0:
            volumeIn_t1 = (volumeIn_t1 - volumeIn_t1.reshape(-1).min()) / (max_d - min_d)
            volumeIn_t2 = (volumeIn_t2 - volumeIn_t2.reshape(-1).min()) / (max_d - min_d)

        volumeDown_t1 = self.degrade(volumeIn_t1).transpose(2, 1, 0)
        volumeDown_t2 = self.degrade(volumeIn_t2).transpose(2, 1, 0)

        if self.opt['phase'] == 'train':
            volumeIn_t1 = volumeIn_t1[:, :, idx]
            volumeIn_t2 = volumeIn_t2[:, :, idx]
            volumeDown_t2 = volumeDown_t2[:, :, idx]
            volumeDown_t1 = volumeDown_t1[:, :, idx]

        volumeDown_t2 = single2tensor3(volumeDown_t2)
        volumeIn_t2 = single2tensor3(volumeIn_t2)
        volumeDown_t1 = single2tensor3(volumeDown_t1)
        volumeIn_t1 = single2tensor3(volumeIn_t1)

        # L:[I_in,ref] H:[HR,HR]
        sample = {'L': [volumeDown_t2, volumeIn_t1],
                  'H': [volumeIn_t2, get_gradient(volumeIn_t2.unsqueeze(1)).squeeze(1)],
                  'path': self.HR_paths[index]}
        return sample

    def __len__(self):
        return len(self.HR_paths)

    def degrade(self, hr_data):
        norm_d = max(hr_data.reshape(-1)) - min(hr_data.reshape(-1))
        if self.scale == 4:
            imgfft = np.fft.fft2(hr_data.transpose(2, 1, 0))
            imgfft = np.fft.fftshift(imgfft)
            imgfft = imgfft[:, 90: 150, 90: 150]
            imgfft = np.fft.ifftshift(imgfft)
            imgifft = np.fft.ifft2(imgfft)
            img_out = abs(imgifft)

        if self.scale == 3:
            imgfft = np.fft.fft2(hr_data.transpose(2, 1, 0))
            imgfft = np.fft.fftshift(imgfft)
            imgfft = imgfft[:, 80: 160, 80: 160]
            imgfft = np.fft.ifftshift(imgfft)
            imgifft = np.fft.ifft2(imgfft)
            img_out = abs(imgifft)

        if self.scale == 2:
            imgfft = np.fft.fft2(hr_data.transpose(2, 1, 0))
            imgfft = np.fft.fftshift(imgfft)
            imgfft = imgfft[:, 60: 180, 60: 180]
            imgfft = np.fft.ifftshift(imgfft)
            imgifft = np.fft.ifft2(imgfft)
            img_out = abs(imgifft)
        if img_out.max() - img_out.min() != 0:
            img_out = (img_out - min(img_out.reshape(-1))) / (max(img_out.reshape(-1)) - min(img_out.reshape(-1)))
        return img_out


class DatasetIXISROriWM(Dataset):
    '''
    Get L/H for SISR.
    must paths_H and path_L both are provided.
    '''

    def __init__(self, opt):
        super(DatasetIXISROriWM, self).__init__()
        self.opt = opt
        self.data_root = opt['dataroot_H']
        self.scale = opt['scale'] if opt['scale'] else 4
        self.HR_paths = get_image_paths(opt['dataroot_H'])
        self.LR_paths = get_image_paths(opt['dataroot_L'])

    def __getitem__(self, index):
        # 或者pd做t1
        volumeDownpath_t2 = self.LR_paths[index].replace('t1', 't2').replace('T1', 'T2').replace('PD', 'T2')
        volumepath_t2 = self.HR_paths[index].replace('t1', 't2').replace('T1', 'T2').replace('PD', 'T2')
        volumeDown_t1 = np.load(self.LR_paths[index])
        volumeDown_t2 = np.load(volumeDownpath_t2)
        volumeIn_t2 = np.load(volumepath_t2)
        volumeIn_t1 = np.load(self.HR_paths[index])
        H, W, C = volumeDown_t2.shape

        idx = random.randint(0, C - 1)
        if self.opt['phase'] == 'train':
            volumeIn_t2 = volumeIn_t2[:, :, idx]
            volumeIn_t1 = volumeIn_t1[:, :, idx]
            volumeDown_t2 = volumeDown_t2[:, :, idx]
            volumeDown_t1 = volumeDown_t1[:, :, idx]

        volumeDown_t2 = single2tensor3(volumeDown_t2)
        volumeIn_t2 = single2tensor3(volumeIn_t2)
        volumeIn_t1 = single2tensor3(volumeIn_t1)
        volumeDown_t1 = single2tensor3(volumeDown_t1)
        # L:[I_in,Rc] H:[HR,HR]
        return {'L': [volumeDown_t2, volumeDown_t1],
                'H': [volumeIn_t2, get_gradient(volumeIn_t2.unsqueeze(1)).squeeze(1)],
                'path': self.HR_paths[index]}

    def __len__(self):
        return len(self.HR_paths)
