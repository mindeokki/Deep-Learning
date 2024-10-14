import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf


def image_to_rate_map(image):
    return spikegen.rate(torch.tensor(image), time_var_input=True)

def image_to_latency_map(image):
    return spikegen.latency(torch.tensor(image), tau=5, clip=True, bypass=True)


if __name__ == '__main__':
    y, sr = sf.read('BD.wav')
    print(y.shape)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=20000, n_fft=2048, hop_length=256)

    # Convert to log scale (dB)
    log_S = librosa.power_to_db(S, ref=np.max)
    print(log_S.shape)

    # Plot the log Mel spectrogram
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel', hop_length=256, fmax=20000, cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Log Mel Spectrogram')
    plt.tight_layout()
    plt.show()

    log_S_normalized = (log_S - np.min(log_S)) / (np.max(log_S) - np.min(log_S))
    print(np.min(log_S_normalized))
    print(np.max(log_S_normalized))

    d = spikegen.rate(torch.tensor(log_S_normalized), time_var_input=True)
    print(d.shape)
    # librosa.display.specshow(d.numpy(), sr=sr, x_axis='time', y_axis='mel', hop_length=256, fmax=20000, cmap='viridis')
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Log Mel Spectrogram')
    # plt.tight_layout()
    # plt.show()
    d_late = image_to_latency_map(torch.tensor(log_S_normalized))
    fig, ax = plt.subplots()
    splt.raster(d_late.T, ax, s=25, c="black")

    plt.title("Input Layer")
    plt.xlabel("Time step")
    plt.ylabel("Neuron Number")
    plt.show()