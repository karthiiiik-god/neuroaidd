import numpy as np
import librosa

def extract_audio_features(y, sr=16000):
    y, _ = librosa.effects.trim(y, top_db=25)
    if y.size == 0:
        return None
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_d = librosa.feature.delta(mfcc)
    mfcc_dd = librosa.feature.delta(mfcc, order=2)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    feat = np.hstack([
        mfcc.mean(axis=1), mfcc.std(axis=1),
        mfcc_d.mean(axis=1), mfcc_dd.mean(axis=1),
        zcr.mean(), zcr.std(), rms.mean(), rms.std(), tempo
    ]).astype(np.float32)
    return feat
