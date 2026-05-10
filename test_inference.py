import numpy as np
from scipy.io import wavfile
import requests
import time
import json

# Create test audio
sample_rate = 16000
duration = 2
t = np.linspace(0, duration, sample_rate * duration)
audio = np.int16(32767 * 0.3 * np.sin(2 * np.pi * 440 * t))
wavfile.write('test_audio.wav', sample_rate, audio)
print('[OK] Audio file created')

# Test inference
print('Starting inference...')
start = time.time()

try:
    with open('test_audio.wav', 'rb') as f:
        files = {'audio': ('test_audio.wav', f, 'audio/wav')}
        data = {'prompt': 'Describe this music:', 'max_new_tokens': '256'}
        response = requests.post('http://127.0.0.1:8000/generate', files=files, data=data, timeout=120)
        response.raise_for_status()
        elapsed = time.time() - start
        print(f'[SUCCESS] Inference in {elapsed:.2f}s')
        print(f'Result: {response.json()}')
except Exception as e:
    print(f'[ERROR] {e}')
