import numpy as np
from src.audio.feature_extractor import extract_audio_features


EMO_LABELS = ["neutral", "stressed", "sad", "angry"]

def predict_emotion(y, sr=16000, speech_clf=None):
    """
    Returns (label, prob) using model if present, else heuristic.
    """
    feat = extract_audio_features(y, sr)
    if feat is None:
        return ("neutral", 0.5)
    if speech_clf is None:
        # very light heuristic
        rms = float(np.sqrt((y**2).mean()))
        tempo = 0.0
        try:
            import librosa
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        except Exception:
            pass
        if rms > 0.12 and tempo > 110:   return ("angry", 0.65)
        if rms > 0.09 and tempo < 90:    return ("stressed", 0.6)
        if rms < 0.04 and tempo < 85:    return ("sad", 0.6)
        return ("neutral", 0.55)
    else:
        # expect sklearn Pipeline with predict_proba; class order EMO_LABELS
        try:
            probs = speech_clf.predict_proba([feat])[0]  # len=4
            idx = int(np.argmax(probs))
            return (EMO_LABELS[idx], float(probs[idx]))
        except Exception:
            return ("neutral", 0.5)
