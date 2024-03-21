import librosa
import numpy as np

def load_audio(audio_path, sr=22050):
    """
    Load an audio file and return the waveform and sample rate.
    """
    waveform, sample_rate = librosa.load(audio_path, sr=sr)
    return waveform, sample_rate

def compute_melspectrogram(waveform, sr=22050, n_fft=2048, hop_length=512, n_mels=128):
    """
    Compute the mel-spectrogram of an audio waveform.
    """
    mel_spectrogram = librosa.feature.melspectrogram(y=waveform, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return mel_spectrogram

def compute_mfcc(waveform, sr=22050, n_mfcc=20):
    """
    Compute the Mel-frequency cepstral coefficients (MFCCs) of an audio waveform.
    """
    mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=n_mfcc)
    return mfcc

def compute_chroma(waveform, sr=22050, n_chroma=12, n_fft=2048, hop_length=512):
    """
    Compute the chromagram of an audio waveform.
    """
    chromagram = librosa.feature.chroma_stft(y=waveform, sr=sr, n_chroma=n_chroma, n_fft=n_fft, hop_length=hop_length)
    return chromagram

def compute_rms(waveform, frame_length=2048, hop_length=512):
    """
    Compute the root mean square (RMS) energy of an audio waveform.
    """
    rms = librosa.feature.rms(y=waveform, frame_length=frame_length, hop_length=hop_length)
    return rms

def compute_spectral_centroid(waveform, sr=22050, n_fft=2048, hop_length=512):
    """
    Compute the spectral centroid of an audio waveform.
    """
    spectral_centroid = librosa.feature.spectral_centroid(y=waveform, sr=sr, n_fft=n_fft, hop_length=hop_length)
    return spectral_centroid

def compute_spectral_bandwidth(waveform, sr=22050, n_fft=2048, hop_length=512):
    """
    Compute the spectral bandwidth of an audio waveform.
    """
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=waveform, sr=sr, n_fft=n_fft, hop_length=hop_length)
    return spectral_bandwidth

def compute_spectral_contrast(waveform, sr=22050, n_fft=2048, hop_length=512, n_bands=6):
    """
    Compute the spectral contrast of an audio waveform.
    """
    spectral_contrast = librosa.feature.spectral_contrast(y=waveform, sr=sr, n_fft=n_fft, hop_length=hop_length, n_bands=n_bands)
    return spectral_contrast

def compute_zero_crossing_rate(waveform, frame_length=2048, hop_length=512):
    """
    Compute the zero-crossing rate of an audio waveform.
    """
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=waveform, frame_length=frame_length, hop_length=hop_length)
    return zero_crossing_rate