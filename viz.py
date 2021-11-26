
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_loss_metrics(path, history):
    fig = plt.figure()
    plt.plot(history['train']['epoch'], history['train']['err'],
             color='b', label='training')
    plt.plot(history['val']['epoch'], history['val']['err'],
             color='c', label='validation')
    plt.legend()
    fig.savefig(os.path.join(path, 'loss.png'), dpi=200)
    plt.close('all')

    fig = plt.figure()
    plt.plot(history['val']['epoch'], history['val']['stft_l2'],
             color='r', label='STFT_l2_distance')
    plt.legend()
    fig.savefig(os.path.join(path, 'metric_stft_l2_distance.png'), dpi=200)
    plt.close('all')

    fig = plt.figure()
    plt.plot(history['val']['epoch'], history['val']['envelope_distance'],
             color='b', label='envelope_distance')
    plt.legend()
    fig.savefig(os.path.join(path, 'metric_envelope_distance.png'), dpi=200)
    plt.close('all')
