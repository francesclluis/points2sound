
import os
import time
import torch
import numpy as np
import random
import scipy.io.wavfile as wavfile
import librosa
from arguments import ArgParser
from dataset import MUSICWaveDataset, MUSICSpecDataset
from models import ModelBuilder
from utils import AverageMeter, save_points, makedirs, center_trim
from viz import plot_loss_metrics
from metrics_binaural import compute_metrics
import MinkowskiEngine as ME


# Network wrapper, defines forward pass
class NetWrapper(torch.nn.Module):
    def __init__(self, nets, crit):
        super(NetWrapper, self).__init__()
        self.net_sound, self.net_vision = nets
        self.crit = crit

    def forward(self, batch_data, args):

        if args.arch_sound == 'demucs':

            in_padded_audio = batch_data['in_padded_audio']
            in_audio_shape = torch.zeros(in_padded_audio.shape[0], in_padded_audio.shape[1], args.audLen)
            gt_audio = batch_data['out_audio']
            feats = batch_data['feats']
            coords = batch_data['coords']

            # 2. forward net_vision
            sin = ME.SparseTensor(feats, coords.int(), allow_duplicate_coords=True) #Create SparseTensor
            visual_feature = self.net_vision.forward(sin)

            # 3. forward audio_vision
            pred_audio = self.net_sound.forward(in_padded_audio, visual_feature)
            pred_audio = torch.squeeze(pred_audio, dim=2)
            pred_audio = center_trim(pred_audio, in_audio_shape)

            # 4. loss
            err = self.crit(pred_audio, gt_audio)

            return err, \
                {'pred_audio': pred_audio}

        elif args.arch_sound == 'unet':
            audio_mix_spec = batch_data['audio_mix_spec']
            audio_diff_spec = batch_data['audio_diff_spec']
            audio_gt = audio_diff_spec[:, :, :-1, :]
            feats = batch_data['feats']
            coords = batch_data['coords']

            # 2. forward net_vision
            sin = ME.SparseTensor(feats, coords.int(), allow_duplicate_coords=True) #Create SparseTensor
            visual_feature = self.net_vision.forward(sin)

            # 3. forward audio_vision
            pred_mask = self.net_sound.forward(audio_mix_spec, visual_feature)

            # complex masking to obtain the predicted spectrogram
            spectrogram_diff_real = audio_mix_spec[:, 0, :-1, :] * pred_mask[:, 0, :, :] - \
                                    audio_mix_spec[:, 1, :-1, :] * pred_mask[:, 1, :, :]
            spectrogram_diff_img = audio_mix_spec[:, 0, :-1, :] * pred_mask[:, 1, :, :] + \
                                   audio_mix_spec[:, 1, :-1, :] * pred_mask[:, 0, :, :]
            pred_diff_audio_spec = torch.cat((spectrogram_diff_real.unsqueeze(1), spectrogram_diff_img.unsqueeze(1)), 1)

            # 4. loss
            err = self.crit(pred_diff_audio_spec, audio_gt)

            return err, \
                {'pred_audio': pred_diff_audio_spec}


# Calculate metrics
def calc_metrics_waveform(batch_data, outputs):
    # meters
    envelope_meter = AverageMeter()
    stft_l2_meter = AverageMeter()

    # fetch data and predictions
    gt_audio = batch_data['out_audio']
    pred_audio = outputs['pred_audio']

    # convert into numpy
    gt_audio = gt_audio.detach().cpu().numpy()
    pred_audio = pred_audio.detach().cpu().numpy()

    # loop over each sample
    B = gt_audio.shape[0]
    for j in range(B):
        gt_audio_j  = gt_audio[j, ...]
        pred_audio_j  = pred_audio[j, ...]

        # binaural performance computes
        stft_l2, envelope_distance = compute_metrics(pred_audio_j, gt_audio_j)

        stft_l2_meter.update(stft_l2)
        envelope_meter.update(envelope_distance)

    return [envelope_meter.average(),
            stft_l2_meter.average()]


def calc_metrics_spectrogram(batch_data, outputs, args):
    # meters
    envelope_meter = AverageMeter()
    stft_l2_meter = AverageMeter()

    # fetch data and predictions
    gt_audio = batch_data['out_audio']
    pred_diff_audio_spec = outputs['pred_audio']
    audio_mix = batch_data['in_audio']

    # convert into numpy
    gt_audio = gt_audio.detach().cpu().numpy()
    pred_diff_audio_spec = pred_diff_audio_spec.detach().cpu().numpy()
    audio_mix = audio_mix.detach().cpu().numpy()

    # loop over each sample
    B = gt_audio.shape[0]
    for j in range(B):
        gt_audio_j = gt_audio[j, ...]
        pred_diff_audio_spec_j = pred_diff_audio_spec[j, ...]
        audio_mix_j= audio_mix[j, ...]

        # ISTFT to convert back to audio
        reconstructed_stft_diff = pred_diff_audio_spec_j[0, :, :] + (1j * pred_diff_audio_spec_j[1, :, :])
        reconstructed_signal_diff = librosa.istft(reconstructed_stft_diff, hop_length=441, win_length=1014, center=True,
                                                  length=args.audLen)
        reconstructed_signal_left = (audio_mix_j + reconstructed_signal_diff) / 2
        reconstructed_signal_right = (audio_mix_j - reconstructed_signal_diff) / 2
        reconstructed_binaural = np.concatenate((reconstructed_signal_left, reconstructed_signal_right), axis=0)

        # binaural performance computes
        stft_l2, envelope_distance = compute_metrics(reconstructed_binaural, gt_audio_j)

        stft_l2_meter.update(stft_l2)
        envelope_meter.update(envelope_distance)

    return [envelope_meter.average(),
            stft_l2_meter.average()]


# Visualize predictions
def output_visuals_waveform(batch_data, outputs, args):
    in_audio = batch_data['in_audio']
    in_padded_audio = batch_data['in_padded_audio']
    gt_audio = batch_data['out_audio']
    features = batch_data['feats']
    coords = batch_data['coords']
    prefix = batch_data['prefix']
    pred_audio = outputs['pred_audio']

    # convert into numpy
    in_audio = in_audio.detach().cpu().numpy()
    in_padded_audio = in_padded_audio.detach().cpu().numpy()
    gt_audio = gt_audio.detach().cpu().numpy()
    pred_audio = pred_audio.detach().cpu().numpy()

    in_audio = np.transpose(in_audio, (0, 2, 1))
    in_padded_audio = np.transpose(in_padded_audio, (0, 2, 1))
    gt_audio = np.transpose(gt_audio, (0, 2, 1))
    pred_audio = np.transpose(pred_audio, (0, 2, 1))

    B = in_audio.shape[0]
    # loop over each sample
    for j in range(B):

        makedirs(os.path.join(args.vis, prefix[j]))

        # save mono
        filename_in_audio = os.path.join(prefix[j], 'in.wav')
        wavfile.write(os.path.join(args.vis, filename_in_audio), args.audRate, in_audio[j, ...])

        filename_in_audio = os.path.join(prefix[j], 'in_padded.wav')
        wavfile.write(os.path.join(args.vis, filename_in_audio), args.audRate, in_padded_audio[j, ...])

        # save output binaural
        filename_gtwav = os.path.join(prefix[j], 'gt.wav')
        filename_predwav = os.path.join(prefix[j], 'pred.wav')
        wavfile.write(os.path.join(args.vis, filename_gtwav), args.audRate, gt_audio[j, ...])
        wavfile.write(os.path.join(args.vis, filename_predwav), args.audRate, pred_audio[j, ...])

        idx = torch.where(coords[:, 0] == j)
        path_point = os.path.join(args.vis, prefix[j], 'point_cloud_scene.ply')
        if args.rgbs_feature:
            colors = np.asarray(features[idx])
            xyz = np.asarray(coords[idx][:, 1:4])
            xyz = xyz*args.voxel_size
            save_points(path_point, xyz, colors)
        else:
            xyz = np.asarray(features[idx])
            save_points(path_point, xyz)

# Visualize predictions
def output_visuals_spectrogram(batch_data, outputs, args):

    in_audio = batch_data['in_audio']
    audio_mix = in_audio
    pred_diff_audio_spec = outputs['pred_audio']
    gt_audio = batch_data['out_audio']

    features = batch_data['feats']
    coords = batch_data['coords']
    prefix = batch_data['prefix']

    # convert into numpy
    in_audio = in_audio.detach().cpu().numpy()
    gt_audio = gt_audio.detach().cpu().numpy()
    pred_diff_audio_spec = pred_diff_audio_spec.detach().cpu().numpy()

    in_audio = np.transpose(in_audio, (0, 2, 1))
    gt_audio = np.transpose(gt_audio, (0, 2, 1))

    B = in_audio.shape[0]

    # loop over each sample
    for j in range(B):

        makedirs(os.path.join(args.vis, prefix[j]))

        # save mono
        filename_in_audio = os.path.join(prefix[j], 'in.wav')
        wavfile.write(os.path.join(args.vis, filename_in_audio), args.audRate, in_audio[j, ...])

        # save output binaural
        filename_gtwav = os.path.join(prefix[j], 'gt.wav')
        wavfile.write(os.path.join(args.vis, filename_gtwav), args.audRate, gt_audio[j, ...])

        #ISTFT to convert pred back to audio
        pred_diff_audio_spec_j = pred_diff_audio_spec[j, ...]
        reconstructed_stft_diff = pred_diff_audio_spec_j[0, :, :] + (1j * pred_diff_audio_spec_j[1, :, :])
        reconstructed_signal_diff = librosa.istft(reconstructed_stft_diff, hop_length=441, win_length=1014, center=True,
                                                  length=args.audLen)
        reconstructed_signal_left = (audio_mix[j, ...] + reconstructed_signal_diff) / 2
        reconstructed_signal_right = (audio_mix[j, ...] - reconstructed_signal_diff) / 2
        pred_audio = np.concatenate((reconstructed_signal_left, reconstructed_signal_right), axis=0)
        pred_audio = np.transpose(pred_audio, (1, 0))

        filename_predwav = os.path.join(prefix[j], 'pred.wav')
        wavfile.write(os.path.join(args.vis, filename_predwav), args.audRate, pred_audio)

        idx = torch.where(coords[:, 0] == j)
        path_point = os.path.join(args.vis, prefix[j], 'point_cloud_scene.ply')
        if args.rgbs_feature:
            colors = np.asarray(features[idx])
            xyz = np.asarray(coords[idx][:, 1:4])
            xyz = xyz*args.voxel_size
            save_points(path_point, xyz, colors)
        else:
            xyz = np.asarray(features[idx])
            save_points(path_point, xyz)


def evaluate(netWrapper, loader, history, epoch, args):
    print('Evaluating at {} epochs...'.format(epoch))
    torch.set_grad_enabled(False)

    # remove previous viz results
    makedirs(args.vis, remove=True)

    # switch to eval mode
    netWrapper.eval()

    # initialize meters
    loss_meter = AverageMeter()
    envelope_meter = AverageMeter()
    stft_l2_meter = AverageMeter()

    for i, batch_data in enumerate(loader):
        # forward pass
        err, outputs = netWrapper.forward(batch_data, args)

        loss_meter.update(err.item())
        print('[Eval] iter {}, loss: {:.4f}'.format(i, err.item()))

        if args.arch_sound == 'demucs':
            # calculate metrics
            envelope_distance, stft_l2 = calc_metrics_waveform(batch_data, outputs)
            # output visualization
            output_visuals_waveform(batch_data, outputs, args)
        if args.arch_sound == 'unet':
            # calculate metrics
            envelope_distance, stft_l2 = calc_metrics_spectrogram(batch_data, outputs, args)
            # output visualization
            output_visuals_spectrogram(batch_data, outputs, args)

        envelope_meter.update(envelope_distance)
        stft_l2_meter.update(stft_l2)

    print('[Eval Summary] Epoch: {}, Loss: {:.4f}, '
          'envelope distance: {:.4f}, stft_l2: {:.4f}'
          .format(epoch, loss_meter.average(),
                  envelope_meter.average(),
                  stft_l2_meter.average()))
    history['val']['epoch'].append(epoch)
    history['val']['err'].append(loss_meter.average())
    history['val']['stft_l2'].append(stft_l2_meter.average())
    history['val']['envelope_distance'].append(envelope_meter.average())

    # Plot figure
    if epoch > 0:
        print('Plotting figures...')
        plot_loss_metrics(args.ckpt, history)


def train(netWrapper, loader, optimizer, history, epoch, args):
    torch.set_grad_enabled(True)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    # switch to train mode
    netWrapper.train()

    # main loop
    torch.cuda.synchronize()
    tic = time.perf_counter()
    for i, batch_data in enumerate(loader):
        # measure data time
        torch.cuda.synchronize()
        data_time.update(time.perf_counter() - tic)

        # forward pass
        netWrapper.zero_grad()
        err, _ = netWrapper.forward(batch_data, args)
        err = err.mean()

        # backward
        err.backward()
        optimizer.step()

        # measure total time
        torch.cuda.synchronize()
        batch_time.update(time.perf_counter() - tic)
        tic = time.perf_counter()

        # display
        if i % args.disp_iter == 0:
            print('Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, '
                  'lr_sound: {}, lr_vision: {}, '
                  'loss: {:.4f}'
                  .format(epoch, i, args.epoch_iters,
                          batch_time.average(), data_time.average(),
                          args.lr_sound, args.lr_vision,
                          err.item()))
            fractional_epoch = epoch - 1 + 1. * i / args.epoch_iters
            history['train']['epoch'].append(fractional_epoch)
            history['train']['err'].append(err.item())


def checkpoint(nets, history, epoch, args):
    print('Saving checkpoints at {} epochs.'.format(epoch))
    (net_sound, net_vision) = nets
    suffix_latest = 'latest.pth'
    suffix_best = 'best.pth'

    torch.save(history,
               '{}/history_{}'.format(args.ckpt, suffix_latest))
    torch.save(net_sound.state_dict(),
               '{}/sound_{}'.format(args.ckpt, suffix_latest))
    torch.save(net_vision.state_dict(),
               '{}/vision_{}'.format(args.ckpt, suffix_latest))

    cur_err = history['val']['err'][-1]
    if cur_err < args.best_err:
        args.best_err = cur_err
        torch.save(net_sound.state_dict(),
                   '{}/sound_{}'.format(args.ckpt, suffix_best))
        torch.save(net_vision.state_dict(),
                   '{}/vision_{}'.format(args.ckpt, suffix_best))


def create_optimizer(nets, args):
    (net_sound, net_vision) = nets
    param_groups = [{'params': net_sound.parameters(), 'lr': args.lr_sound},
                    {'params': net_vision.features.parameters(), 'lr': args.lr_vision},
                    {'params': net_vision.fc.parameters(), 'lr': args.lr_sound}]
    return torch.optim.Adam(param_groups)


def collate_all(list_data):

    # Process samples to form a batch. Point Cloud information is processed to be compatible with Minkowski Engine (ME)
    # Note that ME creates a batch of point clouds coordinates putting the batch indice in the first column
    # Samples come from dataset/music.py or dataset/spec.py

    miscellaneous_data = []

    coords = []
    feats = []

    for sample in list_data:
        miscellaneous_data.append(sample[0])
        coords_sample, points_sample, rgbs_sample, rgbs_feature = sample[-1]

        coords.append(coords_sample)
        if rgbs_feature:
            feats.append(rgbs_sample)
        else:
            feats.append(points_sample)

    coords_batch = ME.utils.batched_coordinates(coords)
    feats_batch = torch.from_numpy(np.vstack(feats)).float()

    batched_dict = torch.utils.data.dataloader.default_collate(miscellaneous_data)

    batched_dict['coords'] = coords_batch
    batched_dict['feats'] = feats_batch

    return batched_dict


def make_data_loader(dset, args):

    args = {
        'batch_size': args.batch_size,
        'num_workers': int(args.workers),
        'collate_fn': collate_all,
        'pin_memory': True,
        'drop_last': False,
        'shuffle': True
    }

    loader = torch.utils.data.DataLoader(dset, **args)

    return loader


def main(args):
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
    crit = builder.build_criterion(arch=args.loss)

    # Dataset and Loader
    if args.arch_sound == 'demucs':

        valid_length = net_sound.valid_length(args.audLen)
        delta = valid_length - args.audLen

        dataset_train = MUSICWaveDataset(
            args.list_train, delta, args, split='train')
        dataset_val = MUSICWaveDataset(
            args.list_val, delta, args, max_sample=args.num_val, split='val')

    elif args.arch_sound == 'unet':
        dataset_train = MUSICSpecDataset(
            args.list_train, args, split='train')
        dataset_val = MUSICSpecDataset(
            args.list_val, args, max_sample=args.num_val, split='val')

    loader_train = make_data_loader(dataset_train, args)
    loader_val = make_data_loader(dataset_val, args)

    args.epoch_iters = len(dataset_train) // args.batch_size
    print('1 Epoch = {} iters'.format(args.epoch_iters))

    # Wrap networks
    netWrapper = NetWrapper(nets, crit)
    netWrapper = torch.nn.DataParallel(netWrapper, device_ids=range(args.num_gpus))
    netWrapper.to(args.device)

    # Set up optimizer
    optimizer = create_optimizer(nets, args)

    # History of performance
    history = {
        'train': {'epoch': [], 'err': []},
        'val': {'epoch': [], 'err': [], 'stft_l2': [], 'envelope_distance': []}}

    # Eval mode
    evaluate(netWrapper, loader_val, history, 0, args)
    if args.mode == 'eval':
        print('Evaluation Done!')
        return

    # Training loop
    for epoch in range(1, args.num_epoch + 1):
        train(netWrapper, loader_train, optimizer, history, epoch, args)

        # Evaluation and visualization
        if epoch % args.eval_epoch == 0:
            evaluate(netWrapper, loader_val, history, epoch, args)

            # checkpointing
            checkpoint(nets, history, epoch, args)

    print('Training Done!')


if __name__ == '__main__':
    # arguments
    parser = ArgParser()
    args = parser.parse_train_arguments()
    args.batch_size = args.num_gpus * args.batch_size_per_gpu
    args.device = torch.device("cuda")

    # experiment name
    if args.mode == 'train':
        args.id += '-{}mix'.format(args.num_mix)
        args.id += '-{}-{}'.format(
            args.arch_vision, args.arch_sound)
        args.id += '-visual_feature_size{}'.format(args.visual_feature_size)
        args.id += '-epoch{}'.format(args.num_epoch)

    print('Model ID: {}'.format(args.id))

    # paths to save/load output
    args.ckpt = os.path.join(args.ckpt, args.id)
    args.vis = os.path.join(args.ckpt, 'visualization/')
    if args.mode == 'train':
        makedirs(args.ckpt, remove=True)
    elif args.mode == 'eval':
        args.weights_sound = os.path.join(args.ckpt, 'sound_best.pth')
        args.weights_vision = os.path.join(args.ckpt, 'vision_best.pth')

    # initialize best error with a big number
    args.best_err = float("inf")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)

