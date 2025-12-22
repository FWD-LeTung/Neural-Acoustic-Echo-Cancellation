import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
file1 = "D:/Downloads/-2jLGNCgf0WDpKMY2iup7g_doubletalk_mic.wav"
file2 = "real2jLG.wav"
file3 = "D:/AEC-Challenge/datasets/synthetic/nearend_speech/nearend_speech_fileid_9109.wav"
# Load audio
y1, sr1 = librosa.load(file1, sr=None)
y2, sr2 = librosa.load(file2, sr=None)
y3, sr3 = librosa.load(file3, sr=None)
# STFT parameters
n_fft = 512
win_length = 320
hop_length = 160

# Compute STFT
S1 = librosa.stft(y1, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
S2 = librosa.stft(y2, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
S3 = librosa.stft(y3, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
# Convert to dB

"""
S3_db = 20 * np.log10(np.abs(S3) + 1e-9)
S1_db = 20 * np.log10(np.abs(S1) + 1e-9)
S2_db = 20 * np.log10(np.abs(S2) + 1e-9)

"""
S1_db = librosa.amplitude_to_db(np.abs(S1), ref=np.max)
S2_db = librosa.amplitude_to_db(np.abs(S2), ref=np.max)
S3_db = librosa.amplitude_to_db(np.abs(S3), ref=np.max)
# Plot
plt.figure(figsize=(10, 12))

# -------- Wave --------
"""
plt.subplot(5, 1, 1)
librosa.display.waveshow(y1, sr=sr1)
plt.title("Mic Signal")

plt.subplot(5, 1, 2)
librosa.display.waveshow(y2, sr=sr2)
plt.title("Estimated")
"""



# -------- STFT --------

plt.subplot(3, 1, 1)
librosa.display.specshow(
    S1_db,
    sr=sr1,
    hop_length=hop_length,
)
plt.title("Mic Signal")

plt.subplot(3, 1, 2)
librosa.display.specshow(
    S2_db,
    sr=sr2,
    hop_length=hop_length,
    
)
plt.title("Estimated")
"""
plt.subplot(3, 1, 3)
librosa.display.specshow(
    S3_db,
    sr=sr3,
    hop_length=hop_length,
    
)
plt.title("Clean")
plt.tight_layout()
"""
plt.savefig("demo2.png")
plt.show()
