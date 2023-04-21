import librosa
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.signal import convolve2d

class AudioSegmentation:
    def __init__(self, audio, sr=None, segment_duration=59, k=1.5, penalize_factor=0.1, kernel_size=10, q=90, adaptive_threshold=0.3, threshold_default=0.15):
        
        if isinstance(input, str):
            self.y, self.sr = librosa.load(audio)
        elif sr is not None:
            self.y = audio
            self.sr = sr
        else:
            raise ValueError("Invalid input: either provide a file path or y and sr values.")

    
        self.segment_duration = segment_duration
        self.k = k # high k -> punishes larger STD
        self.penalize_factor = penalize_factor
        self.kernel_size = kernel_size
        self.q = q
        self.adaptive_threshold = adaptive_threshold
        self.threshold_default = threshold_default

        self.y = librosa.to_mono(self.y)
        self.y = librosa.effects.trim(self.y)[0]
        self.y = librosa.util.normalize(self.y)

        self.binary_result = self.compute_binary_result()
        # self.harmony, y_percussive = librosa.effects.hpss(self.y)

    def compute_cqt(self):
        cqt = librosa.amplitude_to_db(librosa.cqt(y=self.y, sr=self.sr), ref=np.max)
        # cqt = librosa.amplitude_to_db(librosa.cqt(y=self.y, sr=self.sr, hop_length=1024, fmin=440.0, n_bins=64, bins_per_octave=16), ref=np.max)
        return librosa.util.normalize(cqt, axis=0)

    def compute_binary_result(self):
        cqt_norm = self.compute_cqt()
        sim_matrix = np.dot(cqt_norm.T, cqt_norm)

        kernel = np.eye(self.kernel_size)
        conv_result = convolve2d(sim_matrix, kernel, mode='valid')
        threshold = np.percentile(conv_result, self.q)

        return (conv_result > threshold).astype(int)

    def remove_frequency_bands(self):
        # Compute the short-time Fourier transform (STFT) of the audio signal
        stft = librosa.stft(self.y)

        # Sub-bass (20 Hz to 60 Hz), Bass (60 Hz to 250 Hz), and Treble (4000 Hz to 20,000 Hz)
        sub_bass_high_cutoff = 60
        bass_high_cutoff = 250
        treble_low_cutoff = 4000
        treble_high_cutoff = 20000

        # Convert frequency cutoffs to the corresponding STFT bin indices
        sub_bass_high_bin = np.floor(sub_bass_high_cutoff / (self.sr / stft.shape[0])).astype(int)
        bass_high_bin = np.floor(bass_high_cutoff / (self.sr / stft.shape[0])).astype(int)
        treble_low_bin = np.floor(treble_low_cutoff / (self.sr / stft.shape[0])).astype(int)
        treble_high_bin = np.floor(treble_high_cutoff / (self.sr / stft.shape[0])).astype(int)

        # Remove the specified frequency bands by zeroing out the corresponding STFT bins
        stft[:sub_bass_high_bin, :] = 0
        stft[sub_bass_high_bin:bass_high_bin, :] = 0
        stft[treble_low_bin:treble_high_bin, :] = 0

        # Compute the inverse STFT to reconstruct the filtered audio signal
        self.y = librosa.istft(stft)
        return self.y

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
                penalized_chunk_cos = chunk_cos * (1 - (time_interval - start_interval) * self.penalize_factor)
                print(f"time_interval:{time_interval} original:{chunk_cos} pen:{penalized_chunk_cos}")
                cos_list.append((penalized_chunk_cos, chunk_cos, time_interval, columns_per_chunk))

            time_interval += increment

        return max(cos_list) if cos_list else (0, 0, 0.5, self.time_to_binary(0.5))

    def binary_to_time(self, binary):
        return round((self.segment_duration * binary) / self.binary_result.shape[1], 2)

    def time_to_binary(self, time_interval):
        return int((time_interval * self.binary_result.shape[1]) // self.segment_duration)
    
    def compute_adaptive_threshold(self, cosine_sim_mean, cosine_sim_std, cosine_similarities_len, decay_factor=0.5, scaling_factor=0.5):
        
        if cosine_similarities_len == 1:
            # Allow some flexibility when creating a new section
            threshold = self.threshold_default
        else:
            # Aims to create a threshold that dynamically adjusts based on consine similarity and standard deviation

            n = 0.1 # mutiplier from low->high cosine similarity (closer to 0 makes k2 value closer to upper-bound)
            m = 1 # universal mutiplier for consine similarity

            # k - value from 1->2 that amplifies the standard deviation
            k2 = m + (cosine_sim_mean ** n) # non-linear mutiplier for standard deviation (larger for higher cosine similarity)
            k = m + cosine_sim_mean # linear mutipler for standard deviation

            threshold = 1 - ((cosine_sim_mean - (cosine_sim_std*k2)) / cosine_sim_mean )

            # safety checks
            if threshold > 1:
                print(f"DEFAULT threshold (>1):{self.threshold_default}")
                return self.threshold_default


        return threshold
    
    def should_create_new_segment(self, chunk1, binary_result, current_column, columns_per_chunk, x, threshold, cosine_sim_mean):
        for i in range(1, x + 1):
            if current_column + (columns_per_chunk * (i + 1)) >= binary_result.shape[1]:
                break
            tmp_chunk = binary_result[:, current_column + (columns_per_chunk * i): current_column + (columns_per_chunk * (i + 1))]
            tmp_cos = self.cos_sim(chunk1, tmp_chunk)

            if tmp_cos == 0.0 or 1 - (tmp_cos / cosine_sim_mean) <= threshold:
               return False

        return True
    
    def find_largest_cosine_change(self, start, end, segment_duration):
        cos_sim_list = []
        prev_sim = 0

        for t in range(start, end - (segment_duration*2) + 1, segment_duration):
            chunk1 = self.binary_result[:, t:t + segment_duration]
            chunk2 = self.binary_result[:, t + segment_duration:t + (2 * segment_duration)]
            cos_sim = self.cos_sim(chunk1, chunk2)
            tmp_sim = prev_sim-cos_sim
            prev_sim = cos_sim
            cos_sim_list.append((tmp_sim,t + segment_duration))

        print(cos_sim_list)
        return max(cos_sim_list)

    def segment_audio(self):
        start_time = 0
        sections_times = []

        pensim, sim, time_interval, columns_per_chunk = self.find_new_interval(self.binary_result[:, 0:self.time_to_binary(10)])

        l, r = 0, (columns_per_chunk * 2)
        cosine_similarities = [sim]
        std_list = []

        while r + columns_per_chunk < self.binary_result.shape[1]:
            chunk1 = self.binary_result[:, l:l + columns_per_chunk]
            chunk2 = self.binary_result[:, r:r + columns_per_chunk]
            chunk_cos = self.cos_sim(chunk1, chunk2)

            cosine_sim_mean = np.mean(cosine_similarities)
            cosine_sim_std = np.std(cosine_similarities)
            std_list.append(cosine_sim_std)
            
            threshold = self.compute_adaptive_threshold(cosine_sim_mean, cosine_sim_std, len(cosine_similarities))
            print(f"time:{self.binary_to_time(r + columns_per_chunk)} time_interval:{time_interval} chunk_cos:{chunk_cos} cosine_sim_mean:{cosine_sim_mean}")
            print(f"loss:{1 - (chunk_cos / cosine_sim_mean)} threshold:{threshold} std: {cosine_sim_std} std_sum: {np.sum(std_list)} len {len(cosine_similarities)}\n")

            if chunk_cos != 0.0 and 1 - (chunk_cos / cosine_sim_mean) > threshold:
                
                # Double check if the new section is a section
                # if not self.should_create_new_segment(chunk1, self.binary_result, r, columns_per_chunk, 1, threshold, cosine_sim_mean):
                #     print("\n\n\n---------------------REVERSED---------------------\n\n\n")
                #     cosine_similarities.pop()
                #     # [('0:00:00', '0:01:41'), ('0:01:39', '0:01:46')]
                #     # [('0:00:00', '0:00:22'), ('0:00:19', '0:01:41'), ('0:01:40', '0:01:46')]
                #     cosine_similarities.append(chunk_cos)
                #     r += columns_per_chunk
                #     continue

                largest_change, largest_change_time = self.find_largest_cosine_change(r, r+columns_per_chunk, self.time_to_binary(0.5))
                print(f"largest_change:{self.binary_to_time(largest_change_time)} largest_change:{largest_change}")

                r = largest_change_time
                end_time = self.binary_to_time(r)
                sections_times.append((start_time, end_time))
                start_time = end_time

                pensim, sim, time_interval, columns_per_chunk = self.find_new_interval(self.binary_result[:, r:self.binary_result.shape[1]])
                
                l = r
                r += columns_per_chunk * 2
                cosine_similarities = [pensim]
                std_list = []
            else:
                cosine_similarities.append(chunk_cos)
                r += columns_per_chunk

        end_time = self.binary_to_time(r + columns_per_chunk)
        sections_times.append((start_time, end_time))

        return sections_times
