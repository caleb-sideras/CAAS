# Caleb's Adaptive Audio Segmentation (CAAS)

CAAS is an audio segmentation algorithm that iteratively finds the best intervals and patterns in a given audio signal using cosine similarity.

## Features

- Adaptive interval selection based on maximizing cosine similarity
- Robust detection of repeating patterns in the audio signal
- Scalable and modular implementation for easy integration in production environments

## Visualizations

CAAS provides visualizations to help understand the different stages of the audio segmentation process:

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

## Algorithm Overview

1. **Import necessary libraries**: Import `librosa` for audio processing, `numpy` for numerical operations, `sklearn.metrics.pairwise` for cosine similarity, and `scipy.signal` for convolutions.

2. **Load the audio file**: Load the audio file using `librosa.load()`.

3. **Compute the CQT**: Compute the CQT of the audio signal using `librosa.cqt()`, convert it to decibels with `librosa.amplitude_to_db()`, and normalize it along the time axis using `librosa.util.normalize()`.

4. **Compute the self-similarity matrix**: Compute the self-similarity matrix by taking the dot product of the normalized CQT with itself.

5. **Convolve the similarity matrix**: Convolve the similarity matrix with a diagonal kernel of size 10 to emphasize diagonal patterns, and threshold the result at the 90th percentile to create a binary result.

6. **Define the cos_sim() function**: Define the `cos_sim()` function, which computes the cosine similarity between two input arrays.

7. **Define the find_new_interval() function**: Define the `find_new_interval()` function, which takes a binary matrix and finds the best interval and the associated cosine similarity by comparing adjacent non-overlapping chunks in the input matrix.

8. **Define the binary_to_time() and time_to_binary() functions**: Define the `binary_to_time()` and `time_to_binary()` functions to convert between binary column indices and time values.

9. **Initialize variables**: Initialize variables for segment duration, start time, sections times, and the initial interval.

10. **Iterate through the binary result matrix**: Iterate through the binary result matrix, comparing adjacent non-overlapping chunks. If their cosine similarity is significantly different from the mean of previous similarities, mark the current time as the end of the current section and the beginning of a new section. Update the interval and continue.

11. **Append the last section's end time**: Once the iteration is complete, append the last section's end time.

12. **Return the final sections times**: Return the final sections times, which represents the segments of the audio file with similar patterns.

## Potential Flaws and Improvements

1. **Audio Sanitization**: The current algorithm was designed for and performs better on sanitized audio segments, specifically a chunked song with removed vocals, percussion and noise. Future implementations will automate this.

2. **Multichannel audio**: The current implementation only supports mono audio. The algorithm could be extended to support multichannel audio by processing each channel separately and combining the results.

3. **Interval selection**: The algorithm assumes that the best interval is found by maximizing cosine similarity. This may not always produce accurate results. Exploring other similarity measures or refining the criteria for choosing intervals could improve segmentation.

4. **Thresholds**: The algorithm has fixed thresholds for segmentation based on my training data. Future iterations of the algorithm will support adpative thresholds based on time intervals, cosine similarity STD and more.

5. **Error handling**: The code does not have proper error handling in case an issue occurs during the execution of the algorithm.Future iterations will have error handling and informative error messages can make the algorithm more robust and user-friendly.


## Dependencies

- Python 3.x
- Librosa
- NumPy
- SciPy
- scikit-learn
- Matplotlib

## Usage

1. Install the required dependencies:

```bash
pip install librosa numpy scipy scikit-learn matplotlib
```

2. Import the AudioSegmentation class and create an instance with your audio file and the desired segment duration:

```python
from caas import AudioSegmentation

audio_file = 'your_audio_file.wav'
segment_duration = 60

audio_segmentation = AudioSegmentation(audio_file, segment_duration)
```

3. Run the segmentation algorithm and print the sections' start and end times:

```python
sections_times = audio_segmentation.segment_audio()
print(sections_times)
```

4. Generate and display visualizations for each step of the algorithm:

```python
audio_segmentation.plot_all_visualizations()
```

## Visualizations

1. Initialization

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

## Notes

This algorithm serves as a minor component of a web application I'm developing called "MusicGPT." MusicGPT is a chatbot designed to enable users to inquire about various aspects of a song, from lyrics to intricate musical terminology. I decided to open-source the segmentation algorithm because I believe it offers a unique yet straightforward solution to segmentation. If you're interested in MusicGPT's development, please feel free to contact me.

License
MIT License


In this README.md, we've included a brief description of the algorithm, its features, visualizations, dependencies, usage instructions, and licensing information. Replace the image file names (e.g., `waveform.png`, `cqt.png`, etc.) with the actual file paths of your visualization images. You can generate these visualization images using the provided plotting functions, and save them using `plt.savefig()` before calling `plt.show()`.