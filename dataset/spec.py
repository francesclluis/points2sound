
import os
import random
import numpy as np
from .base import BaseDataset


class MUSICSpecDataset(BaseDataset):
    def __init__(self, list_sample, opt, **kwargs):
        super(MUSICSpecDataset, self).__init__(
            list_sample, opt, **kwargs)
        self.max_mix = opt.num_mix
        self.rgbs_feature = opt.rgbs_feature
        self.angles = [0, 45, 90, 135, 180, 225, 270, 315]

    def __getitem__(self, index):
        N = np.random.randint(1, self.max_mix+1)
        points = [None for n in range(N)]
        rgbs = [None for n in range(N)]
        binaural_audios = [None for n in range(N)]
        infos = [[] for n in range(N)]
        path_pointclouds = ['' for n in range(N)]
        path_audios_binaural = ['' for n in range(N)]

        instruments = []
        infos[0] = self.list_sample[index]
        path_instr = self.list_sample[index][0]
        instr = os.path.basename(os.path.dirname(path_instr))
        instruments.append(instr)

        while len(instruments)!= N:
            indexN = random.randint(0, len(self.list_sample)-1)
            path_instr = self.list_sample[indexN][0]
            instr = os.path.basename(os.path.dirname(path_instr))
            if instr not in instruments:
                infos[len(instruments)] = self.list_sample[indexN]
                instruments.append(instr)

        angles = random.sample(self.angles, N)

        # select point cloud musician
        for n, infoN in enumerate(infos):
            path_audioN, path_pointcloudsN, count_pointcloudsN = infoN
            pointcloudN = random.randint(0, int(count_pointcloudsN)-1)
            path_pointclouds[n] = os.path.join(path_pointcloudsN, '{:05d}.ply'.format(pointcloudN))
            path_audios_binaural[n] = path_audioN+'_'+str(angles[n])+'.wav'

        # load frames and audios
        for n, infoN in enumerate(infos):
            points[n], rgbs[n] = self._load_pointclouds(path_pointclouds[n], angles[n])
            binaural_audios[n] = self._load_bin_audio(path_audios_binaural[n])
        #Create audio mixtures
        output_audio_mix = self._mix_n(binaural_audios)
        input_audio_mix = output_audio_mix[0, :] + output_audio_mix[1, :]

        audio_mix_spec = self.generate_spectrogram(input_audio_mix)
        audio_diff_spec = self.generate_spectrogram(output_audio_mix[0, :]-output_audio_mix[1, :])

        input_audio_mix = np.expand_dims(input_audio_mix, axis=0)

        #create 3D scene
        scene_points = np.concatenate(points)
        scene_rgbs = np.concatenate(rgbs)
        # create 3D scene coords
        scene_coords = self._create_coords(scene_points)

        ret_dict = {'audio_mix_spec': audio_mix_spec, 'audio_diff_spec': audio_diff_spec}
        if self.split != 'train':

            prefix = []
            for n in range(N):
                prefix.append('-'.join(infos[n][0].split('/')[-2:]).split('.')[0])
            prefix = '+'.join(prefix)

            ret_dict['in_audio'] = input_audio_mix
            ret_dict['out_audio'] = output_audio_mix
            ret_dict['prefix'] = prefix
            ret_dict['n_sources'] = N

        return ret_dict, (scene_coords, scene_points, scene_rgbs, self.rgbs_feature)
