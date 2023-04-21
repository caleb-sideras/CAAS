# Caleb's Adaptive Audio Segmentation (CAAS)

CAAS is an audio segmentation algorithm that iteratively finds the best intervals and patterns in a given audio signal, clusters them based on similarity and then extracts relavent audio features (Lyrics, MIDI etc).

## Features

- Adaptive interval selection based on maximizing cosine similarity
- Concatination and clustering of like segments
- Feature extraction - Lyrics, MIDI etc

## Dependencies

- Python 3.x
- Librosa
- NumPy
- SciPy
- scikit-learn
- Matplotlib
- Soundfile

## Usage

1. Install the required dependencies:

```bash
pip install librosa numpy scipy scikit-learn matplotlib soundfile
```

2. Configure your path to vamp plugins in AudioToMidi.py
```python
os.environ['VAMP_PATH'] = 'path/to/conda/envs/your_environment_name/vamp_plugins'
```

3. Import the AudioProcessor class and create an instance with your input/ouput directory, vocal boolean and OpenAI API key:

```python
from AudioProcessor import AudioProcessor

audio_directory = 'songs'
has_vocals = True
out_directory = 'out'
openai_api_key = 'sk-...'

ap = AudioProcessor(audio_directory, has_vocals, out_directory, openai_api_key)
```

4. Run the AudioProcessor algorithm and print the cluster data and json file path:

```python
audio_file = 'OliverTree-MissYou' # Currently only supports .wav

cluster_data, json_filepath = ap.process_audio(audio_file)
print(cluster_data, json_filepath)
```

5. See the output directory for MIDI, WAV and JSON data:

Due to DMCA restrictions, I am unable to upload the segments that were generated from the song, but trust me... these segments/clusters are VERY similar. Listen to them and tell me i'm wrong!!!

Example JSON data:
```json
{
    "name": "oliver",
    "tempo": "143.5546875",
    "clusters": [
        [
            {
                "start": "0.0",
                "end": "26.16",
                "lyrics": "Don't remind me I'm minding more than first breath Don't try to find me I'm just a woman in light It's just a surprise for me Don't pretend you don't know I'm the only one you know It could be anything you say\n",
                "midi": [
                    {
                        "time": "0.74",
                        "duration": "0.14",
                        "value": "78.0"
                    },
                    ...
                    {
                        "time": "25.99",
                        "duration": "0.17",
                        "value": "64.0"
                    }
                ],
                "mfccs_mean": [
                    "-43.892063",
                    ...
                    "-2.6905072"
                ],
                "spectral_contrast_mean": [
                    "20.418052966303513",
                    ...
                    "48.16223226793056"
                ],
                "chroma_cqt_mean": [
                    "0.2101367",
                    ...
                    "0.29208136"
                ]
            },
            {
                "start": "159.11",
                "end": "169.05"
            },
            {
                "start": "79.47",
                "end": "100.8"
            }
        ],
        [
            {
                "start": "100.8",
                "end": "120.23",
                "lyrics": "I'm not angry, but if somebody else, it could be anyone else, I would say, I'm super mad. I'm super mad. What's that? Where is the connection? Who's next?\n",
                "midi": [
                    {
                        "time": "0.53",
                        "duration": "0.4",
                        "value": "68.0"
                    },
                    ...
                    {
                        "time": "19.22",
                        "duration": "0.13",
                        "value": "73.0"
                    }
                ],
                "mfccs_mean": [
                    "-21.953321",
                    ...
                    "-3.1610284"
                ],
                "spectral_contrast_mean": [
                    "22.204617377186477",
                    ...
                    "48.983418602856524"
                ],
                "chroma_cqt_mean": [
                    "0.24937618",
                    ...
                    "0.43595913"
                ]
            },
            {
                "start": "26.16",
                "end": "40.1"
            }
        ],
        [
            {
                "start": "40.1",
                "end": "79.47",
                "lyrics": "I don't know if it's true, but I love you so Love me one more time, I'm gonna love you more I don't know if it's true, but I love you so Love me one more time, I'm gonna love you more\n",
                "midi": [
                    {
                        "time": "0.57",
                        "duration": "0.21",
                        "value": "66.0"
                    },
                    ...
                    {
                        "time": "37.66",
                        "duration": "1.7",
                        "value": "64.0"
                    }
                ],
                "mfccs_mean": [
                    "17.616764",
                    ...
                    "-0.042170133"
                ],
                "spectral_contrast_mean": [
                    "21.81925076941477",
                    ...
                    "49.20934462420385"
                ],
                "chroma_cqt_mean": [
                    "0.46816224",
                    ...
                    "0.5819297"
                ]
            },
            {
                "start": "120.23",
                "end": "159.11"
            }
        ]
    ]
}
```

## Visualizations

CAAS provides visualizations to help understand the different stages of the initial audio segmentation process:

1. **Waveform**: Displays the waveform of the input audio signal.

![Waveform](figures/waveform.png)

2. **Constant-Q Transform (CQT)**: Displays the CQT of the input audio signal, which represents the frequency content of the signal over time.

![CQT](figures/cqt.png)

3. **Similarity Matrix**: Displays the similarity between different segments of the audio signal based on cosine similarity.

![Similarity Matrix](figures/similarity_matrix.png)

4. **Binary Result Matrix**: Displays the binary result matrix obtained after applying the convolution operation on the similarity matrix with a kernel matrix.

![Binary Result with Convolution Kernel](figures/binary_result.png)

5. **Segmentation**: Displays the segmentation of the audio signal, with the red dashed lines indicating the start and end times of each segment.

![Segmentation](figures/segmentation.png)

## Code for Visualizations

1. Initialization of AudioSegmentation class

```python
from caas import AudioSegmentation
import Visualizations

audio_file = 'your_audio_file.wav'
segment_duration = 60

audio_segmentation = AudioSegmentation(audio_file, segment_duration)
```

2. Plotting

```python
# Plotting Waveform
plot_waveform(audio_segmentation.y, audio_segmentation.sr)

# Plotting CQT
cqt = audio_segmentation.compute_cqt()
plot_cqt(cqt)

# Plotting Similarity Matrix
sim_matrix = np.dot(cqt.T, cqt)
plot_similarity_matrix(sim_matrix)

# Plotting Binary Result
plot_binary_result(audio_segmentation.binary_result)

# Plotting Binary Result with Convolution Kernel
kernel = np.eye(10)
plot_binary_result_with_kernel(audio_segmentation.binary_result, kernel)

# Plotting Segmentation
sections_times = audio_segmentation.segment_audio()
plot_segmentation(audio_segmentation, audio_segmentation.binary_result, sections_times, segment_duration)

# Show all plots
plt.show()
```

## Potential Flaws and Improvements

1. **Multichannel audio**: The current implementation only supports mono audio. The algorithm could be extended to support multichannel audio by processing each channel separately and combining the results.

2. **Interval selection**: The algorithm assumes that the best interval is found by maximizing cosine similarity. This may not always produce accurate results. Exploring other similarity measures or refining the criteria for choosing intervals could improve segmentation.

3. **Error handling**: The code does not have proper error handling in case an issue occurs during the execution of the algorithm.Future iterations will have error handling and informative error messages can make the algorithm more robust and user-friendly.

4. **Transcription**: Lyric extraction is super janky using OpenAI's Whisper model, as it's designed for regular speech. Open to alternate suggestions. 

## Notes

This algorithm serves as a minor component of a web application I'm developing called "MusicGPT." MusicGPT is a chatbot designed to enable users to inquire about various aspects of a song, from lyrics to intricate musical terminology. I decided to open-source the segmentation algorithm because I believe it offers a unique yet straightforward solution to segmentation. If you're interested in MusicGPT's development, please feel free to contact me.

To get spleeter running with Python 3, clone the repo in the CAAS directory and change useless_config.json to the following:

```json
{
    "mix_name": "mix",
    "instrument_list": [
        "vocals",
        "other"
    ],
    "sample_rate": 44100,
    "frame_length": 4096,
    "frame_step": 1024,
    "T": 128,
    "F": 128,
    "n_channels": 2,
    "chunk_duration": 4,
    "n_chunks_per_song": 1,
    "separation_exponent": 2,
    "mask_extension": "zeros",
    "learning_rate": 0.0001,
    "batch_size": 2,
    "train_max_steps": 10,
    "throttle_secs": 20,
    "save_checkpoints_steps": 100,
    "save_summary_steps": 5,
    "random_seed": 0,
    "model": {
        "type": "unet.unet",
        "params": {
            "conv_activation": "ELU",
            "deconv_activation": "ELU"
        }
    },
    "train_csv": "/tmp/tmppkwgulux/train/train.csv",
    "validation_csv": "/tmp/tmppkwgulux/train/train.csv",
    "model_dir": "/tmp/tmppkwgulux/model_2",
    "training_cache": "/tmp/tmppkwgulux/cache_2/training",
    "validation_cache": "/tmp/tmppkwgulux/cache_2/validation"
}
```

License
MIT License