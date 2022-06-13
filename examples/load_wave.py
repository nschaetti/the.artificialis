
# Imports
import wave
import math
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

# Load wav file
samplerate, data = wavfile.read('session.wav')

print(samplerate)
print(data)

plt.plot(data[:500])
plt.show()
