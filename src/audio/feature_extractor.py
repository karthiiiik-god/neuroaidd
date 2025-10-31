# src/audio/feature_extractor.py
# Minimal, stable audio feature builder for NeuroAid

from __future__ import annotations
import numpy as np

try:
    import librosa
except Exception:
    librosa = None


def _safe(val, default=0.0):
    try:
        if np.isnan(val) or np.isinf(val):
            return default
        return float(val)
    except Exception:
        return default


def extract_audio_features(y: np.ndarray, sr: int = 16000) -> np.ndarray | None:
    """
    Returns a 1D np.ndarray of features or None if feature extraction isn't possible.
    Features are robust and fixed-length:
      - MFCC mean/std (20 + 20)
      - Delta/Delta-Delta mean (20 + 20)
      - ZCR mean/std (2)
      - RMS mean/std (2)
      - Tempo (1)
    Total: 85 floats
    """
    if y is None:
        return None
    y = np.asarray(y, dtype=np.float32).squeeze()
    if y.size == 0:
        return None
    if librosa is None:
        # Fallback: energy + zero-cross rate only (still returns fixed length)
        rms = _safe(np.sqrt((y ** 2).mean()))
        zcr = _safe(np.mean(np.abs(np.diff(np.sign(y))) > 0))
        # 85-dim placeholder with first two meaningful, rest zeros
        feat = np.zeros(85, dtype=np.float32)
        feat[0] = rms
        feat[1] = zcr
        return feat

    # Trim leading/trailing silence
    y, _ = librosa.effects.trim(y, top_db=25)

    if y.size == 0:
        return None

    # Core features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)                 # (20, T)
    mfcc_d = librosa.feature.delta(mfcc)                               # (20, T)
    mfcc_dd = librosa.feature.delta(mfcc, order=2)                     # (20, T)
    zcr = librosa.feature.zero_crossing_rate(y)                        # (1, T)
    rms = librosa.feature.rms(y=y)                                     # (1, T)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    # Aggregate
    parts = [
        mfcc.mean(axis=1), mfcc.std(axis=1),
        mfcc_d.mean(axis=1), mfcc_dd.mean(axis=1),
        [zcr.mean(), zcr.std(), rms.mean(), rms.std(), tempo]
    ]
    feat = np.hstack(parts).astype(np.float32)

    # Guard NaNs/Infs
    feat = np.array([_safe(v) for v in feat], dtype=np.float32)

    # Ensure exactly 85 dims (20+20+20+20+5)
    if feat.shape[0] != 85:
        pad = np.zeros(85, dtype=np.float32)
        pad[:min(85, feat.shape[0])] = feat[:85]
        feat = pad
    return feat


__all__ = ["extract_audio_features"]
