
import numpy as np
import librosa
from scipy.signal import hilbert


def STFT_l2_distance(predicted_binaural, gt_binaural):
    # channel1
    predicted_spect_channel1 = librosa.core.stft(np.asfortranarray(predicted_binaural[0, :]), n_fft=1024, hop_length=441, win_length=1014, center=True)
    gt_spect_channel1 = librosa.core.stft(np.asfortranarray(gt_binaural[0, :]), n_fft=1024, hop_length=441, win_length=1014, center=True)
    real = np.expand_dims(np.real(predicted_spect_channel1), axis=0)
    imag = np.expand_dims(np.imag(predicted_spect_channel1), axis=0)
    predicted_realimag_channel1 = np.concatenate((real, imag), axis=0)
    real = np.expand_dims(np.real(gt_spect_channel1), axis=0)
    imag = np.expand_dims(np.imag(gt_spect_channel1), axis=0)
    gt_realimag_channel1 = np.concatenate((real, imag), axis=0)
    channel1_distance = np.mean(np.power((predicted_realimag_channel1 - gt_realimag_channel1), 2))

    # channel2
    predicted_spect_channel2 = librosa.core.stft(np.asfortranarray(predicted_binaural[1, :]), n_fft=1024, hop_length=441, win_length=1014, center=True)
    gt_spect_channel2 = librosa.core.stft(np.asfortranarray(gt_binaural[1, :]), n_fft=1024, hop_length=441, win_length=1014, center=True)
    real = np.expand_dims(np.real(predicted_spect_channel2), axis=0)
    imag = np.expand_dims(np.imag(predicted_spect_channel2), axis=0)
    predicted_realimag_channel2 = np.concatenate((real, imag), axis=0)
    real = np.expand_dims(np.real(gt_spect_channel2), axis=0)
    imag = np.expand_dims(np.imag(gt_spect_channel2), axis=0)
    gt_realimag_channel2 = np.concatenate((real, imag), axis=0)
    channel2_distance = np.mean(np.power((predicted_realimag_channel2 - gt_realimag_channel2), 2))

    # sum the distance between two channels
    stft_l2_distance = channel1_distance + channel2_distance
    return float(stft_l2_distance)


def Envelope_distance(predicted_binaural, gt_binaural):
    # channel1
    pred_env_channel1 = np.abs(hilbert(predicted_binaural[0, :]))
    gt_env_channel1 = np.abs(hilbert(gt_binaural[0, :]))
    channel1_distance = np.sqrt(np.mean((gt_env_channel1 - pred_env_channel1)**2))

    # channel2
    pred_env_channel2 = np.abs(hilbert(predicted_binaural[1, :]))
    gt_env_channel2 = np.abs(hilbert(gt_binaural[1, :]))
    channel2_distance = np.sqrt(np.mean((gt_env_channel2 - pred_env_channel2)**2))

    # sum the distance between two channels
    envelope_distance = channel1_distance + channel2_distance
    return float(envelope_distance)


def compute_metrics(predicted_binaural, gt_binaural):
    stft_l2 = STFT_l2_distance(predicted_binaural, gt_binaural)
    envelope_distance = Envelope_distance(predicted_binaural, gt_binaural)
    return stft_l2, envelope_distance
