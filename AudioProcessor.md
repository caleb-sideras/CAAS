# AudioProcessor README

The AudioProcessor class is a high-level interface for processing audio files. It is designed to analyze and segment audio files, transcribe vocals, and convert the accompaniment to MIDI files. This class makes use of several libraries including librosa, spleeter, and AudioToMidiMelodia.

## Features

- Segment audio files into clusters based on their features.
- Transcribe vocals in the audio files using OpenAI's Whisper ASR.
- Extract vocal and accompaniment parts from the audio files using spleeter.
- Convert the accompaniment part to a MIDI file using the AudioToMidiMelodia algorithm.
- Extract features such as MFCCs, spectral contrast, and chroma from the audio signal.

## How the algorithm works

The AudioProcessor class follows these steps:

1. Initialize the instance with the audio directory, an option to extract vocals or not, output directory, and the OpenAI API key (if needed).
2. `process_audio()` method is called for a specific audio file.
3. The AudioCluster class segments the audio file into different clusters based on the audio features. These clusters represent different sections of the audio file.
4. For the longest segment in each cluster, the following steps are performed:
    - If vocals=True, the audio file is separated into vocals and accompaniment using spleeter. The vocals are transcribed using OpenAI's Whisper ASR.
    - The accompaniment is converted into a MIDI file using the AudioToMidiMelodia algorithm.
    - MFCCs, spectral contrast, and chroma features are extracted using the librosa library.
5. All extracted data and features are saved as a JSON file in the output directory.

## Example usage

```python
audio_processor = AudioProcessor(audio_directory='path/to/audio_directory', vocals=True, out_dir='output_directory', api_key='your_openai_api_key')
audio_data, json_filepath = audio_processor.process_audio('audio_file_name')
```

## Dependencies
- openai
- soundfile
- librosa
- numpy
- AudioCluster
- spleeter
- AudioToMidiMelodia