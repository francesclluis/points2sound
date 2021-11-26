
import os
import torch
import numpy as np
import open3d as o3d
import scipy.io.wavfile as wavfile
import librosa
import MinkowskiEngine as ME
from arguments import ArgParser
from models import ModelBuilder


def generate_spectrogram(audio):
    spectro = librosa.core.stft(audio, n_fft=1024, hop_length=441, win_length=1014, center=True)
    real = np.expand_dims(np.real(spectro), axis=0)
    imag = np.expand_dims(np.imag(spectro), axis=0)
    spectro_two_channel = np.concatenate((real, imag), axis=0)
    return spectro_two_channel


# Network wrapper, defines forward pass
class NetWrapper(torch.nn.Module):
    def __init__(self, nets):
        super(NetWrapper, self).__init__()
        self.net_sound, self.net_vision = nets

    def forward(self, in_data, args, predicted_binaural_spec_path):

        hop_size = 0.05
        audio = in_data['audio']
        feats = in_data['feats']
        feats = torch.from_numpy(feats).to(args.device)

        coords = in_data['coords']
        coords = torch.from_numpy(coords).to(args.device)
        coords = ME.utils.batched_coordinates([coords])

        # 1.forward net_vision
        sin = ME.SparseTensor(feats.float(), coords.int(), allow_duplicate_coords=True)
        visual_feature = self.net_vision.forward(sin)

        overlap_count = np.zeros((2, audio.shape[1])) #count the number of times a data point is calculated
        binaural_audio = np.zeros((2, audio.shape[1]))

        # perform spatialization over the whole spectrogram in a siliding-window fashion
        sliding_window_start = 0
        samples_per_window = args.audLen

        while sliding_window_start + samples_per_window < audio.shape[-1]:
            sliding_window_end = sliding_window_start + samples_per_window
            audio_segment_mix = audio[:, sliding_window_start:sliding_window_end]
            audio_mix_spec = torch.FloatTensor(generate_spectrogram(np.squeeze(audio_segment_mix))).unsqueeze(0).to(args.device)

            # 3. forward audio_vision
            pred_mask = self.net_sound.forward(audio_mix_spec, visual_feature)

            # complex masking to obtain the predicted spectrogram
            spectrogram_diff_real = audio_mix_spec[:, 0, :-1, :] * pred_mask[:, 0, :, :] - \
                                    audio_mix_spec[:, 1, :-1, :] * pred_mask[:, 1, :, :]
            spectrogram_diff_img = audio_mix_spec[:, 0, :-1, :] * pred_mask[:, 1, :, :] + \
                                   audio_mix_spec[:, 1, :-1, :] * pred_mask[:, 0, :, :]
            pred_diff_audio_spec = torch.cat((spectrogram_diff_real.unsqueeze(1), spectrogram_diff_img.unsqueeze(1)), 1)
            pred_diff_audio_spec = pred_diff_audio_spec.detach().cpu().numpy()

            # ISTFT to convert back to audio
            pred_diff_audio_spec_j = pred_diff_audio_spec[0, ...]
            reconstructed_stft_diff = pred_diff_audio_spec_j[0, :, :] + (1j * pred_diff_audio_spec_j[1, :, :])
            reconstructed_signal_diff = librosa.istft(reconstructed_stft_diff, hop_length=441, win_length=1014, center=True, length=args.audLen)
            reconstructed_signal_left = (audio_segment_mix + reconstructed_signal_diff) / 2
            reconstructed_signal_right = (audio_segment_mix - reconstructed_signal_diff) / 2
            reconstructed_binaural = np.concatenate((reconstructed_signal_left, reconstructed_signal_right), axis=0)

            binaural_audio[:, sliding_window_start:sliding_window_end] = binaural_audio[:, sliding_window_start:sliding_window_end] + reconstructed_binaural
            overlap_count[:, sliding_window_start:sliding_window_end] = overlap_count[:, sliding_window_start:sliding_window_end] + 1
            sliding_window_start = sliding_window_start + int(hop_size * args.audRate)

        # deal with the last segment
        audio_segment_mix = audio[:, -samples_per_window:]
        audio_mix_spec = torch.FloatTensor(generate_spectrogram(np.squeeze(audio_segment_mix))).unsqueeze(0).to(args.device)
        pred_mask = self.net_sound.forward(audio_mix_spec, visual_feature)

        # complex masking to obtain the predicted spectrogram
        spectrogram_diff_real = audio_mix_spec[:, 0, :-1, :] * pred_mask[:, 0, :, :] - \
                                audio_mix_spec[:, 1, :-1, :] * pred_mask[:, 1, :, :]
        spectrogram_diff_img = audio_mix_spec[:, 0, :-1, :] * pred_mask[:, 1, :, :] \
                               + audio_mix_spec[:, 1, :-1, :] * pred_mask[:, 0, :, :]
        pred_diff_audio_spec = torch.cat((spectrogram_diff_real.unsqueeze(1), spectrogram_diff_img.unsqueeze(1)), 1)
        pred_diff_audio_spec = pred_diff_audio_spec.detach().cpu().numpy()

        # ISTFT to convert back to audio
        pred_diff_audio_spec_j = pred_diff_audio_spec[0, ...]
        reconstructed_stft_diff = pred_diff_audio_spec_j[0,:,:] + (1j * pred_diff_audio_spec_j[1,:,:])
        reconstructed_signal_diff = librosa.istft(reconstructed_stft_diff, hop_length=441, win_length=1014, center=True, length=args.audLen)
        reconstructed_signal_left = (audio_segment_mix + reconstructed_signal_diff) / 2
        reconstructed_signal_right = (audio_segment_mix - reconstructed_signal_diff) / 2
        reconstructed_binaural = np.concatenate((reconstructed_signal_left, reconstructed_signal_right), axis=0)

        # add the spatialized audio to reconstructed_binaural
        binaural_audio[:,-samples_per_window:] = binaural_audio[:,-samples_per_window:] + reconstructed_binaural
        overlap_count[:,-samples_per_window:] = overlap_count[:,-samples_per_window:] + 1

        # divide aggregated predicted audio by their corresponding counts
        binaural_audio = np.divide(binaural_audio, overlap_count)
        binaural_audio = np.transpose(binaural_audio, (1, 0))

        wavfile.write(predicted_binaural_spec_path, args.audRate, binaural_audio)


def main(args, nets, song_path, scene_3d_path, predicted_binaural_spec_path, rgbs_feature):
    #Load song
    gt_audio, rate = librosa.load(song_path, sr=args.audRate, mono=False)
    in_audio = gt_audio[0, :] + gt_audio[1, :]
    audio = np.expand_dims(in_audio, axis=0)

    #Load 3D scene
    pcd = o3d.io.read_point_cloud(scene_3d_path)
    points = np.array(pcd.points)
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
    netWrapper.forward(in_data, args, predicted_binaural_spec_path)


parser = ArgParser()
args = parser.parse_train_arguments()
args.device = torch.device("cuda")

test_data_path = os.path.join(os.path.split(os.path.join(os.getcwd(), __file__))[0], 'small_test_data', 'Source_N_'+str(args.num_mix))

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

for folder in folders:
    test_folder_path = os.path.join(test_data_path, folder)
    print(test_folder_path)
    song_path = os.path.join(test_folder_path, 'gt.wav')
    scene_3d_path = os.path.join(test_folder_path, 'gt_point_cloud_scene.ply')
    rgbs_feature = True
    predicted_binaural_spec_path = os.path.join(test_folder_path, 'pred_Mono2Binaural_rgbs.wav')
    main(args, nets, song_path, scene_3d_path, predicted_binaural_spec_path, rgbs_feature)
