import matplotlib.pyplot as plt
import librosa
import numpy as np

def plot_waveform(y, sr):
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title('Waveform')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.tight_layout()

def plot_cqt(cqt):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(cqt, x_axis='time', y_axis='cqt_note')
    plt.title('Constant-Q Transform (CQT)')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()

def plot_similarity_matrix(sim_matrix):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(sim_matrix, x_axis='time', y_axis='time')
    plt.title('Similarity Matrix')
    plt.xlabel('Time Segment')
    plt.ylabel('Time Segment')
    plt.colorbar()
    plt.tight_layout()

def plot_binary_result(binary_result):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(binary_result, x_axis='time', y_axis='time')
    plt.title('Binary Result')
    plt.xlabel('Time Segment')
    plt.ylabel('Time Segment')
    plt.colorbar()
    plt.tight_layout()

def plot_segmentation(audio_segmentation, binary_result, sections_times, segment_duration):
    plt.figure(figsize=(10, 4))
    plt.imshow(binary_result, aspect='auto', cmap='gray_r', origin='lower')
    plt.title('Segmentation')
    plt.xlabel('Time Segment')
    plt.ylabel('Time Segment')

    for start, end in sections_times:
        start_bin = audio_segmentation.time_to_binary(start)
        end_bin = audio_segmentation.time_to_binary(end)
        plt.axvline(x=start_bin, color='r', linestyle='--')
        plt.axvline(x=end_bin, color='r', linestyle='--')

    plt.xticks(np.arange(0, binary_result.shape[1], audio_segmentation.time_to_binary(10)), np.arange(0, segment_duration, 10))
    plt.tight_layout()

def plot_convolution_kernel(kernel):
    plt.figure(figsize=(2, 2))
    plt.imshow(kernel, cmap='gray_r', origin='lower')
    plt.title('Convolution Kernel')
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.xticks(range(kernel.shape[1]))
    plt.yticks(range(kernel.shape[0]))
    plt.tight_layout()

def plot_binary_result_with_kernel(binary_result, kernel):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={'width_ratios': [5, 1]})

    # Plot binary result
    im1 = ax1.imshow(binary_result, aspect='auto', cmap='gray_r', origin='lower')
    ax1.set_title('Binary Result')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Frequency Bin')
    fig.colorbar(im1, ax=ax1)

    # Plot convolution kernel
    im2 = ax2.imshow(kernel, cmap='gray_r', origin='lower')
    ax2.set_title('Convolution Kernel')
    ax2.set_xlabel('Column')
    ax2.set_ylabel('Row')
    ax2.set_xticks(range(kernel.shape[1]))
    ax2.set_yticks(range(kernel.shape[0]))

    # Add arrow and label
    ax1.annotate('Convolution', xy=(0.8, 0.5), xytext=(0.5, 0.5), xycoords='axes fraction', textcoords='axes fraction',
                 arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12, ha='center')

    plt.tight_layout()

def plot_segments(binary_result, segments, song_length):
    segment_length = 30
    num_segments_30s = song_length // segment_length + 1

    for i in range(num_segments_30s):
        start_time_30s = i * segment_length
        end_time_30s = min((i + 1) * segment_length, song_length)
        start_binary_30s = int((start_time_30s * binary_result.shape[1]) // song_length)
        end_binary_30s = int((end_time_30s * binary_result.shape[1]) // song_length)

        binary_result_30s = binary_result[:, start_binary_30s:end_binary_30s]

        fig, ax = plt.subplots(figsize=(15, 5))
        ax.imshow(binary_result_30s, aspect='auto', cmap='gray_r')

        for start_time, end_time in segments:
            start_binary = int((start_time * binary_result.shape[1]) // song_length)
            end_binary = int((end_time * binary_result.shape[1]) // song_length)

            if start_time_30s <= start_time < end_time_30s:
                ax.axvline(start_binary - start_binary_30s, linestyle='--', color='red')
            if start_time_30s <= end_time < end_time_30s:
                ax.axvline(end_binary - start_binary_30s, linestyle='--', color='red')

        ax.set_xticks(np.linspace(0, binary_result_30s.shape[1], 6))
        ax.set_xticklabels(np.round(np.linspace(start_time_30s, end_time_30s, 6), 2))
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency Bin')

        plt.show(block=False)

    input("Press Enter to close all windows...")
    plt.close('all')
