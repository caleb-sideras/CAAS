# AudioSegmentation README

This Python script defines an AudioSegmentation class that takes an audio file or an audio waveform as input and segments the audio into sections based on the similarity between consecutive segments.

The AudioSegmentation class works by analyzing the audio input and segmenting it into sections based on similarity. Here's an overview of how the algorithm works in steps:

1. **Load and preprocess the audio**: The input audio is loaded using librosa. If a file path is provided, the audio is loaded directly. If a waveform is provided, the sampling rate must also be given. The audio is then converted to mono, trimmed, and normalized.
2. **Compute the Constant-Q Transform (CQT)**: The CQT of the preprocessed audio is calculated, which is a time-frequency representation that provides a more perceptually meaningful representation of the audio. The CQT is then normalized along the frequency axis.
3. **Compute the binary result**: The similarity matrix is calculated by computing the dot product of the normalized CQT with itself. A 2D convolution is then performed on the similarity matrix with an identity kernel, and a threshold is applied to create a binary result matrix.
4. **Find the new interval**: The algorithm calculates the cosine similarity between different chunks of the binary result matrix to find the time interval that results in the highest penalized cosine similarity.
5. **Compute the adaptive threshold**: Based on the mean and standard deviation of the cosine similarities computed so far, the algorithm calculates a threshold that adjusts dynamically.
6. **Determine if a new segment should be created**: The algorithm checks if the cosine similarity between two consecutive chunks is below the adaptive threshold, which indicates that a new segment should be created.
7. **Find the largest cosine change**: If a new segment is to be created, the algorithm finds the largest change in cosine similarity within the current segment to determine the exact time to create the new segment.
8. **Segment the audio**: The start and end times of the segments are recorded in a list of tuples. The process is repeated until the entire audio has been analyzed and segmented.
9. **Output the segmented audio**: The algorithm returns a list of tuples representing the start and end times of each audio segment.

## Dependencies

The script requires the following libraries:

- librosa
- numpy
- sklearn.metrics.pairwise
- scipy.signal

## Usage

To use the AudioSegmentation class, first import the necessary libraries and the class:

```python
import librosa
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.signal import convolve2d
from audio_segmentation import AudioSegmentation
```

Next, create an AudioSegmentation object by providing the required input parameters:

```python
audio_segmentation = AudioSegmentation(
    audio, sr=None, segment_duration=59, k=1.5, penalize_factor=0.1,
    kernel_size=10, q=90, adaptive_threshold=0.3, threshold_default=0.15)
The input parameters are as follows:
```

**audio**: The audio file path or a numpy array containing the audio waveform.
sr: The sampling rate of the audio waveform. Required if the input is a numpy array.
**segment_duration**: The duration of each segment in seconds. Default is 59 seconds.
**k**: A parameter that affects the penalization of larger standard deviations. Default is 1.5.
**penalize_factor**: A factor that affects the penalization of larger time intervals. Default is 0.1.
**kernel_size**: The size of the kernel used in the 2D convolution. Default is 10.
**q**: The percentile value used for threshold