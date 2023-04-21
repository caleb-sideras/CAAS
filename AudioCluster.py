import librosa
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import collections
import math
from AudioSegmentation import AudioSegmentation

class AudioCluster:
    def __init__(self, audio_file):
        self.audio_file = audio_file
        self.y, self.sr = librosa.load(audio_file)
        self.segment_duration = math.floor(librosa.get_duration(y=self.y, sr=self.sr))
        self.audio_segmentation = AudioSegmentation(audio=self.y, sr=self.sr, segment_duration=self.segment_duration)
        self.new_y = self.audio_segmentation.remove_frequency_bands()
        self.sections_times = self.audio_segmentation.segment_audio()
        self.features = []
        self.clusters = collections.defaultdict(list)

    def getSr(self):
        return self.sr
    
    def getY(self):
        return self.y
    
    def getTempo(self):
        return librosa.beat.tempo(y=self.y, sr=self.sr)
    
    def compute_features(self):
        for start, end in self.sections_times:
            y_section = self.y[int(start * self.sr):int(end * self.sr)]
            mfcc = librosa.feature.mfcc(y=y_section, sr=self.sr, n_mfcc=13)
            mfcc_mean = np.mean(mfcc, axis=1)
            self.features.append(mfcc_mean)

    def normalize_features(self):
        scaler = StandardScaler()
        self.features_normalized = scaler.fit_transform(self.features)

    def compute_similarity_matrix(self):
        self.similarity_matrix = cosine_similarity(self.features_normalized)

    def perform_clustering(self, num_clusters):
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(self.features_normalized)
        self.section_clusters = kmeans.labels_

    def assign_sections_to_clusters(self):
        for i in range(len(self.section_clusters)):
            self.clusters[self.section_clusters[i]].append(self.sections_times[i])

    @staticmethod
    def combine_sections(segments):
        combined_segments = []
        for segment in segments:
            if not combined_segments or abs(combined_segments[-1][2] - segment[0]) > 0.01:
                combined_segments.append((round(segment[1] - segment[0], 2), *segment))
            else:
                start_time, end_time = combined_segments[-1][1:3]
                new_end_time = segment[1]
                combined_segments[-1] = (round(new_end_time - start_time, 2), start_time, new_end_time)
        return combined_segments

    def process_clusters(self, num_clusters):
        self.compute_features()
        self.normalize_features()
        self.compute_similarity_matrix()
        self.perform_clustering(num_clusters)
        self.assign_sections_to_clusters()

        new_clusters = {}

        for key, cluster_segments in self.clusters.items():
            combined_cluster_segments = self.combine_sections(cluster_segments)
            new_clusters[key] = combined_cluster_segments

        return new_clusters
