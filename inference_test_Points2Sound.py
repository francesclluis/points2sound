
import os
import torch
import numpy as np
import open3d as o3d
import scipy.io.wavfile as wavfile
import librosa
import MinkowskiEngine as ME
from arguments import ArgParser
from models import ModelBuilder
from utils import center_trim
from scipy.linalg import expm, norm


# Network wrapper, defines forward pass
class NetWrapper(torch.nn.Module):
    def __init__(self, nets):
        super(NetWrapper, self).__init__()
        self.net_sound, self.net_vision = nets

    def forward(self, in_data, args, predicted_binaural_waveform_path):

        in_audio = in_data['audio']

        feats = in_data['feats']
        feats = torch.from_numpy(feats).to(args.device)

        coords = in_data['coords']
        coords = torch.from_numpy(coords).to(args.device)
        coords = ME.utils.batched_coordinates([coords])

        # 1.forward net_vision
        sin = ME.SparseTensor(feats.float(), coords.int(), allow_duplicate_coords=True)
        visual_feature = self.net_vision.forward(sin)

        # 2. forward audio_vision
        valid_length = self.net_sound.valid_length(args.audLen)
        delta = valid_length - args.audLen
        in_padded_audio = np.pad(in_audio, ((0, 0), (delta//2, delta - delta//2)))
        in_padded_audio = np.expand_dims(in_padded_audio, axis=0)
        in_padded_audio = torch.from_numpy(in_padded_audio).to(args.device)

        pred_audio = self.net_sound.forward(in_padded_audio, visual_feature)

        pred_audio = torch.squeeze(pred_audio, dim=2)
        in_audio = torch.from_numpy(np.expand_dims(in_audio, axis=0)).to(args.device)
        pred_audio = center_trim(pred_audio, in_audio)

        pred_audio = pred_audio.detach().cpu().numpy()
        pred_audio = pred_audio[0, ...]
        binaural_audio = np.transpose(pred_audio, (1, 0))

        wavfile.write(predicted_binaural_waveform_path, args.audRate, binaural_audio)


def main(args, nets, song_path, scene_3d_path, predicted_binaural_waveform_path, rgbs_feature, rotate):
    # Load song
    gt_audio, rate = librosa.load(song_path, sr=args.audRate, mono=False)
    in_audio = gt_audio[0, :] + gt_audio[1, :]
    audio = np.expand_dims(in_audio, axis=0)

    # Load 3D scene
    pcd = o3d.io.read_point_cloud(scene_3d_path)
    points = np.array(pcd.points)
    if rotate:
        angle = np.pi/2
        axis = np.array([0, 1, 0])
        R = expm(np.cross(np.eye(3), axis / norm(axis) * -angle))
        points = points @ R
    colors = np.array(pcd.colors)
    coords = np.floor(points/args.voxel_size)
    if rgbs_feature:
        feats = colors
    else:
        feats = points

    in_data = {'audio': audio, 'feats': feats, 'coords': coords}

    netWrapper = NetWrapper(nets)
    netWrapper = torch.nn.DataParallel(netWrapper, device_ids=range(args.num_gpus))
    netWrapper.to(args.device)
    netWrapper.eval()
    netWrapper.forward(in_data, args, predicted_binaural_waveform_path)

parser = ArgParser()
args = parser.parse_train_arguments()
args.device = torch.device("cuda")

test_data_path = os.path.join(os.path.split(os.path.join(os.getcwd(), __file__))[0], 'small_test_data',
                              'Source_N_'+str(args.num_mix))

# paths to save/load output
args.ckpt = os.path.join(args.ckpt, args.id)
if args.mode == 'inference':
    args.weights_sound = os.path.join(args.ckpt, 'sound_best.pth')
    args.weights_vision = os.path.join(args.ckpt, 'vision_best.pth')

# Network Builders
builder = ModelBuilder()
net_sound = builder.build_sound(
    arch=args.arch_sound,
    visual_feature_size=args.visual_feature_size,
    weights=args.weights_sound)
net_vision = builder.build_vision(
    arch=args.arch_vision,
    visual_feature_size=args.visual_feature_size,
    weights=args.weights_vision)
nets = (net_sound, net_vision)

folders = os.listdir(test_data_path)

rotate = False
for i in range(2):
    if i:
        rotate = True
    for folder in folders:
        test_folder_path = os.path.join(test_data_path, folder)
        print(test_folder_path)
        song_path = os.path.join(test_folder_path, 'gt.wav')
        scene_3d_path = os.path.join(test_folder_path, 'gt_point_cloud_scene.ply')
        rgbs_feature = True
        if rotate:
            predicted_binaural_waveform_path = os.path.join(test_folder_path, 'pred_rotate_Points2Sound_rgbs.wav')
        else:
            predicted_binaural_waveform_path = os.path.join(test_folder_path, 'pred_Points2Sound_rgbs.wav')
        main(args, nets, song_path, scene_3d_path, predicted_binaural_waveform_path, rgbs_feature, rotate)
