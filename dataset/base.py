try:
    import open3d as o3d
except ImportError:
    raise ImportError('Please install open3d with `pip install open3d`.')
import random
import csv
import numpy as np
import torch
import torch.utils.data as torchdata
from torchvision import transforms
import librosa
from scipy.linalg import expm, norm
from . import point_transforms as ptransforms


class BaseDataset(torchdata.Dataset):
    def __init__(self, list_sample, opt, max_sample=-1, split='train'):
        self.voxel_size = opt.voxel_size
        self.audRate = opt.audRate
        self.audLen = opt.audLen
        self.rgbs_feature = opt.rgbs_feature
        self.split = split
        self.seed = opt.seed
        random.seed(self.seed)

        # initialize point cloud transforms
        self._init_ptransform()

        # list_sample can be a python list or a csv file of list
        if isinstance(list_sample, str):
            self.list_sample = []
            for row in csv.reader(open(list_sample, 'r'), delimiter=','):
                if len(row) < 2:
                    continue
                self.list_sample.append(row)
        elif isinstance(list_sample, list):
            self.list_sample = list_sample
        else:
            raise('Error list_sample!')

        if self.split == 'train':
            self.list_sample *= opt.dup_trainset
            random.shuffle(self.list_sample)

        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]

        num_sample = len(self.list_sample)
        assert num_sample > 0
        print('# samples: {}'.format(num_sample))

    def __len__(self):
        return len(self.list_sample)

    def _init_ptransform(self):
        transform_list = []
        color_transform_list = []
        if self.split == 'train':
            transform_list.append(ptransforms.RandomTranslation(0.15))
            transform_list.append(ptransforms.RandomShear())
            color_transform_list.append(ptransforms.RandomGaussianNoise())
            color_transform_list.append(ptransforms.RandomValue())
            color_transform_list.append(ptransforms.RandomSaturation())
        else:
            pass #apply no transformation in evaluation mode
        self.point_transform = transforms.Compose(transform_list)
        self.color_transform = transforms.Compose(color_transform_list)

    def _create_coords(self, points):
        return np.floor(points/self.voxel_size)

    def _load_pointclouds(self, path, angle):
        points, rgbs = self._load_pointcloud(path)
        angle = angle * np.pi/180
        t = [-np.sin(angle), 0, -np.cos(angle)]
        points = self._rotate(points, angle)
        points = self._translate(points, t=t, max_factor=2)
        points = self.point_transform(points)
        if self.rgbs_feature:
            rgbs = self.color_transform(rgbs)
        return points, rgbs

    def _load_pointcloud(self, path):
        pcd = o3d.io.read_point_cloud(path)
        points = np.array(pcd.points)
        colors = np.array(pcd.colors)
        return points, colors

    def _rotate(self, xyz, angle, axis=np.array([0, 1, 0])):
        R = expm(np.cross(np.eye(3), axis / norm(axis) * -angle))
        return xyz @ R

    def _translate(self, xyz, t=None, max_factor=None):
        return xyz + t*(1+max_factor*np.random.rand(1))

    def _load_bin_audio(self, path_binaural):

        audio_raw_binaural, rate = librosa.load(path_binaural, sr=None, mono=False)
        len_raw = audio_raw_binaural.shape[1]
        offset = np.random.randint(0, len_raw - self.audLen)
        audio_binaural = audio_raw_binaural[:, offset:offset+self.audLen]

        return audio_binaural

    def _mix_n(self, audios):
        N = len(audios)
        # mix
        for n in range(N):
            audios[n] /= N
        audio_mix = np.asarray(audios).sum(axis=0)

        for n in range(N):
            audios[n] = torch.from_numpy(audios[n])

        return audio_mix

    def generate_spectrogram(self, audio):
        spectro = librosa.core.stft(audio, n_fft=1024, hop_length=441, win_length=1014, center=True)
        real = np.expand_dims(np.real(spectro), axis=0)
        imag = np.expand_dims(np.imag(spectro), axis=0)
        spectro_two_channel = np.concatenate((real, imag), axis=0)
        return spectro_two_channel
