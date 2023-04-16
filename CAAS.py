import librosa.display
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.signal import convolve2d

class AudioSegmentation:
    def __init__(self, audio_path, segment_duration=59):
        self.audio_path = audio_path
        self.segment_duration = segment_duration
        self.y, self.sr = librosa.load(audio_path)
        self.binary_result = self.compute_binary_result()

    def compute_cqt(self):
        cqt = librosa.amplitude_to_db(librosa.cqt(y=self.y, sr=self.sr), ref=np.max)
        return librosa.util.normalize(cqt, axis=0)

    def compute_binary_result(self):
        cqt_norm = self.compute_cqt()
        sim_matrix = np.dot(cqt_norm.T, cqt_norm)

        kernel_size = 10
        kernel = np.eye(kernel_size)
        conv_result = convolve2d(sim_matrix, kernel, mode='valid')
        threshold = np.percentile(conv_result, 90)

        return (conv_result > threshold).astype(int)

    @staticmethod
    def cos_sim(array1, array2):
        return cosine_similarity(array1.reshape(1, -1), array2.reshape(1, -1))[0][0]

    def find_new_interval(self, new_binary, start_interval=1, max_interval=5, increment=0.5):
        cos_list = []
        total_columns = new_binary.shape[1]
        time_interval = start_interval
        columns_per_chunk = 0
        while time_interval <= max_interval and columns_per_chunk < total_columns:
            columns_per_chunk = self.time_to_binary(time_interval)

            if columns_per_chunk * 2 <= new_binary.shape[1]:
                chunk1 = new_binary[:, 0:columns_per_chunk]
                chunk2 = new_binary[:, columns_per_chunk:columns_per_chunk * 2]
                chunk_cos = self.cos_sim(chunk1, chunk2)
                cos_list.append((chunk_cos, time_interval, columns_per_chunk))

            time_interval += increment

        return max(cos_list)

    def binary_to_time(self, binary):
        return round((self.segment_duration * binary) / self.binary_result.shape[1], 2)

    def time_to_binary(self, time_interval):
        return int((time_interval * self.binary_result.shape[1]) // self.segment_duration)

    def segment_audio(self):
        start_time = 0
        sections_times = []

        sim, time_interval, columns_per_chunk = self.find_new_interval(self.binary_result[:, 0:self.time_to_binary(10)])

        l, r = 0, (columns_per_chunk * 2)
        cosine_similarities = [sim]

        while r + columns_per_chunk < self.binary_result.shape[1]:
            chunk1 = self.binary_result[:, l:l + columns_per_chunk]
            chunk2 = self.binary_result[:, r:r + columns_per_chunk]
            chunk_cos = self.cos_sim(chunk1, chunk2)

            cosine_sim_mean = np.mean(cosine_similarities)
            cosine_sim_std = np.std(cosine_similarities)

            if chunk_cos != 0.0 and 1 - (chunk_cos / cosine_sim_mean) > 0.3:
                end_time = self.binary_to_time(r + columns_per_chunk)
                sections_times.append((start_time, end_time))
                start_time = end_time
                r += columns_per_chunk
                sim, time_interval, columns_per_chunk = self.find_new_interval(self.binary_result[:, r:self.binary_result.shape[1]])
                l = r
                r += columns_per_chunk * 2
                cosine_similarities = [sim]
            else:
                cosine_similarities.append(chunk_cos)
                r += columns_per_chunk

        end_time = self.binary_to_time(r + columns_per_chunk)
        sections_times.append((start_time, end_time))

        return sections_times
