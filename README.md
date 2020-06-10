# Sound Texture Synthesis

Implements several neural Sound Texture Synthesis methods using PyTorch.
This repository still needs cleaning up and proper documentation to be 
useful to others. This was a Machine Learning Practical (University of Edinburgh)
course project.

Models implemented follow "Synthesizing Diverse, High-Quality Audio Textures" 
by Antognini et al. (2018) and "Sound Texture Synthesis using Convolutional Neural Networks"
by Caracalla & Röbel (2019).

The best approach implemented adopts the Autocorrelation Loss from Antognini
to the time-domain synthesis using CNNs by Caracalla & Röbel.

# Requirements

* PyTorch 1.3
* Librosa

TODO: List additional dependencies with their required versions. 

# Normalisation
To prepare data for listening tests, we need to first normalise the all audio files to the
same intensity.  For this, we use `ffmpeg-normalize`. Install it with `sudo apt install
ffmpeg-normalize`, and run:

```
	ffmpeg-normalize -ofmt wav -ext wav -ar 16000 *.wav
```

in a folder with the required wav files. A new folder (called normalized) will be created
that includes those files. `-ofmt wav -ext wav` makes sure that the output is saved as
.wav files, and `-ar 16000` sets the output bitrate to 16kHz.
