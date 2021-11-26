
import os
import librosa
import argparse
import statistics as stat
import numpy as np
from metrics_binaural import compute_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_mix', type=int, required=True)
    args = parser.parse_args()

    spec_stft_distance_list = []
    spec_envelope_distance_list = []
    wave_stft_distance_list = []
    wave_envelope_distance_list = []
    mono_stft_distance_list = []
    mono_envelope_distance_list = []
    shift_stft_distance_list = []
    shift_envelope_distance_list = []

    num_mix = args.num_mix
    test_data_path = os.path.join(os.path.split(os.path.join(os.getcwd(), __file__))[0], 'small_test_data',
                                  'Source_N_'+str(num_mix))
    folders = os.listdir(test_data_path)

    for folder in folders:

        #define paths
        predicted_binaural_spec_path = os.path.join(test_data_path, folder, 'pred_Mono2Binaural_rgbs.wav')
        predicted_binaural_wave_path = os.path.join(test_data_path, folder, 'pred_Points2Sound_rgbs.wav')
        gt_binaural_path = os.path.join(test_data_path, folder, 'gt.wav')
        shift_path = os.path.join(test_data_path, folder, 'pred_rotate_Points2Sound_rgbs.wav')
        print(gt_binaural_path)

        #load audios
        spec_predicted_binaural, _ = librosa.load(predicted_binaural_spec_path, sr=44100, mono=False)
        wave_predicted_binaural, _ = librosa.load(predicted_binaural_wave_path, sr=44100, mono=False)
        gt_audio_binaural, _ = librosa.load(gt_binaural_path, sr=44100, mono=False)
        mono_audio = gt_audio_binaural[0, :] + gt_audio_binaural[1, :]
        mono_audio = np.repeat(np.expand_dims(mono_audio, 0), 2, axis=0)
        shift_audio, _ = librosa.load(shift_path, sr=44100, mono=False)

        stft_l2_spec, envelope_distance_spec = compute_metrics(spec_predicted_binaural, gt_audio_binaural)
        spec_stft_distance_list.append(stft_l2_spec)
        spec_envelope_distance_list.append(envelope_distance_spec)

        stft_l2_wave, envelope_distance_wave = compute_metrics(wave_predicted_binaural, gt_audio_binaural)
        wave_stft_distance_list.append(stft_l2_wave)
        wave_envelope_distance_list.append(envelope_distance_wave)

        stft_l2_mono, envelope_distance_mono = compute_metrics(mono_audio, gt_audio_binaural)
        mono_stft_distance_list.append(stft_l2_mono)
        mono_envelope_distance_list.append(envelope_distance_mono)

        stft_l2_shift, envelope_distance_shift = compute_metrics(shift_audio, gt_audio_binaural)
        shift_stft_distance_list.append(stft_l2_shift)
        shift_envelope_distance_list.append(envelope_distance_shift)


    print("MONO2BINAURAL STFT L2 Distance: ", stat.mean(spec_stft_distance_list), stat.stdev(spec_stft_distance_list), stat.stdev(spec_stft_distance_list) / np.sqrt(len(spec_stft_distance_list)))
    print("MONO2BINAURAL Average Envelope Distance: ", stat.mean(spec_envelope_distance_list), stat.stdev(spec_envelope_distance_list), stat.stdev(spec_envelope_distance_list) / np.sqrt(len(spec_envelope_distance_list)))

    print("POINTS2SOUND STFT L2 Distance: ", stat.mean(wave_stft_distance_list), stat.stdev(wave_stft_distance_list), stat.stdev(wave_stft_distance_list) / np.sqrt(len(wave_stft_distance_list)))
    print("POINTS2SOUND Average Envelope Distance: ", stat.mean(wave_envelope_distance_list), stat.stdev(wave_envelope_distance_list), stat.stdev(wave_envelope_distance_list) / np.sqrt(len(wave_envelope_distance_list)))

    print("MONO-MONO STFT L2 Distance: ", stat.mean(mono_stft_distance_list), stat.stdev(mono_stft_distance_list), stat.stdev(mono_stft_distance_list) / np.sqrt(len(mono_stft_distance_list)))
    print("MONO-MONO Average Envelope Distance: ", stat.mean(mono_envelope_distance_list), stat.stdev(mono_envelope_distance_list), stat.stdev(mono_envelope_distance_list) / np.sqrt(len(mono_envelope_distance_list)))

    print("ROTATED STFT L2 Distance: ", stat.mean(shift_stft_distance_list), stat.stdev(shift_stft_distance_list), stat.stdev(shift_stft_distance_list) / np.sqrt(len(shift_stft_distance_list)))
    print("ROTATED Average Envelope Distance: ", stat.mean(shift_envelope_distance_list), stat.stdev(shift_envelope_distance_list), stat.stdev(shift_envelope_distance_list) / np.sqrt(len(shift_envelope_distance_list)))


if __name__ == '__main__':
    main()
