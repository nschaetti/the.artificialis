

# Imports
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# ArtGAN imports
from theartificialis.utils import write_curve

# Sample rate
sample_rate = 44100
duration = 10
frequency = 100
frequency2 = 50
a = 0.5
b = 0.5
theta = 0

# Time value
time = np.arange(0, duration, 1.0 / sample_rate)

# Create sinus
audio_curve1 = a * np.sin(2.0 * np.pi * frequency * time + theta)
audio_curve2 = b * np.sin(2.0 * np.pi * frequency2 * time + theta)
audio_curve = audio_curve1 + audio_curve2

plt.plot(audio_curve[:int(sample_rate)])
plt.show()

# write_curve("test.wav", audio_curve, sample_rate)
wavfile.write('test.wav', sample_rate, audio_curve)
