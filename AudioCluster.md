# AudioCluster README

The AudioCluster class takes an audio file as input and segments it into sections based on similarity. Then, it clusters these segments to group similar sections together.

## How the algorithm works

1. **Load the audio**: The input audio file is loaded using librosa. The audio waveform and the sampling rate are stored as instance variables.
2. **Segment the audio**: The AudioSegmentation class is used to segment the audio into sections based on similarity. The start and end times of the sections are stored as instance variables.
3. **Compute MFCC features**: For each audio segment, the MFCC (Mel-frequency cepstral coefficients) features are computed using librosa.
4. **Normalize the features**: The computed MFCC features are normalized using the StandardScaler from sklearn.
5. **Compute similarity matrix**: The cosine similarity matrix between the normalized MFCC features is computed.
6. **Perform clustering**: KMeans clustering from sklearn is used to cluster the segments based on their normalized MFCC features.
7. **Assign sections to clusters**: Each segment is assigned to a cluster based on the KMeans clustering results.
8. **Combine continuous sections**: For each cluster, continuous sections are combined to form larger segments.
9. **Process the clusters**: This function performs the above steps to cluster the audio segments and return the processed clusters.

## Usage

To use the AudioCluster class, first import the necessary libraries and the class:

```python
import librosa
import numpy as np
from AudioSegmentation import AudioSegmentation
from AudioCluster import AudioCluster
```

Next, create an AudioCluster object by providing the required input parameters:

```python
audio_cluster = AudioCluster(audio_file='path/to/audio_file')
```

Then, process the clusters by providing the desired number of clusters:

```python
num_clusters = 3
new_clusters = audio_cluster.process_clusters(num_clusters)
The new_clusters variable will contain a dictionary with cluster labels as keys and a list of combined audio segments as values.
```

## Dependencies
The class requires the following libraries:

- librosa
- numpy
- sklearn.metrics.pairwise
- sklearn.cluster
- sklearn.preprocessing
- collections
- math
- AudioSegmentation
