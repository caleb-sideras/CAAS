import os
import json
import openai
import soundfile as sf
import librosa
import numpy as np
from AudioCluster import AudioCluster
from spleeter.separator import Separator
from spleeter.audio.adapter import AudioAdapter
from AudioToMidi import AudioToMidiMelodia


class AudioProcessor:
    def __init__(self, audio_directory, vocals=False, out_dir='out', api_key=None):
        self.audio_directory = audio_directory
        self.vocals = vocals
        self.out_dir = out_dir
        self.separator = Separator('spleeter:2stems')
        self.audio_loader = AudioAdapter.default()
        self.audio_to_midi = AudioToMidiMelodia(0.25, 0.1)

        if api_key is not None:
            openai.api_key = api_key

    def transcribe_audio(self, file_path):
        audio_file = open(file_path, "rb")
        try:
            transcript = openai.Audio.transcribe("whisper-1", audio_file, response_format='text', language='en')
        except Exception as e:
            raise Exception("An error occurred: {}".format(str(e)))

        return transcript
    
    def extract_features(self, y, sr):
        return {
            'mfccs_mean': [str(x) for x in np.mean(librosa.feature.mfcc(y=y, sr=sr), axis=1)],
            'spectral_contrast_mean': [str(x) for x in np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)],
            'chroma_cqt_mean': [str(x) for x in np.mean(librosa.feature.chroma_cqt(y=y, sr=sr), axis=1)]
        }

    def process_audio(self, audio_file):
        audio_path = os.path.join(self.audio_directory, f"{audio_file}.wav")

        audio_cluster = AudioCluster(audio_path)
        new_clusters = audio_cluster.process_clusters(num_clusters=3)

        sr = audio_cluster.getSr()
        y = audio_cluster.getY()
        tempo = audio_cluster.getTempo()

        cluster_data = {'name': audio_file, 'tempo': str(tempo.tolist()[0])}
        cluster_data['clusters'] = []

        for cluster, segments in new_clusters.items():
            cluster_list = []
            sorted_list = sorted(segments)
            duration, start, end = map(float, segments[-1])
            remaining_segments = sorted_list[:-1]

            current_y = y[round(start) * sr:round(end) * sr]

            current_cluster = {
                'start': str(start),
                'end': str(end),
            }

            cluster_dir = os.path.join(self.out_dir, audio_file)
            os.makedirs(cluster_dir, exist_ok=True)

            if self.vocals:
                waveform, _ = self.audio_loader.load(audio_path, sample_rate=sr, offset=start, duration=duration)
                prediction = self.separator.separate(waveform)

                cluster_filepath_vocals = os.path.join(cluster_dir, f'vocals_{cluster}.wav')
                sf.write(cluster_filepath_vocals, prediction['vocals'], sr)
                transcription = self.transcribe_audio(cluster_filepath_vocals)
                current_cluster["lyrics"] = transcription

                cluster_filepath_accompaniment = os.path.join(cluster_dir, f'accompaniment_{cluster}.wav')
                sf.write(cluster_filepath_accompaniment, prediction['accompaniment'], sr)
                midi_filepath = os.path.join(cluster_dir, f'accompaniment_{cluster}.mid')
                data = self.audio_to_midi.audio_to_midi_melodia(cluster_filepath_accompaniment, tempo, outfile=midi_filepath, savejams=False, return_format=1)
                current_cluster["midi"] = data

            else:
                midi_filepath = os.path.join(cluster_dir, f'accompaniment_{cluster}.mid')
                data = self.audio_to_midi.audio_to_midi_melodia(current_y, tempo, sr=sr, outfile=midi_filepath, savejams=False, return_format=1)
                current_cluster["midi"] = data

            current_cluster.update(self.extract_features(current_y, sr))            
            cluster_list.append(current_cluster)
            
            for duration, start, end in remaining_segments:
                cluster_list.append({
                    'start': str(start),
                    'end' : str(end),
                })

            cluster_data['clusters'].append(cluster_list)

        json_file = f"{audio_file}_clusters.json"
        json_filepath = os.path.join(cluster_dir, json_file)
        with open(json_filepath, "w") as file:
            json.dump(cluster_data, file)

        return (cluster_data, json_filepath)


