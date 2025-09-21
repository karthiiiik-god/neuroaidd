# train_speech_baseline.py — trains SVM from data/speech/{calm,stress}
import os, glob, pickle, numpy as np, librosa
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from src.audio.feature_extractor import extract_audio_features

DATA_ROOT = "data/speech"
OUT_PATH  = "src/models/speech_model.pkl"
SR = 16000
EXTS = (".wav", ".mp3", ".flac", ".m4a", ".ogg")

def list_files(d):
    out=[]; [out.extend(glob.glob(os.path.join(d, f"*{e}"))) for e in EXTS]
    return out

def load_data():
    X, y = [], []
    for label, yval in [("calm",0), ("stress",1)]:
        folder = os.path.join(DATA_ROOT, label)
        files = list_files(folder)
        for p in files:
            try:
                sig, sr = librosa.load(p, sr=SR, mono=True)
                feat = extract_audio_features(sig, SR)
                if feat is None: continue
                X.append(feat); y.append(yval)
            except Exception as e:
                print("[WARN] skip", p, e)
    if not X:
        raise RuntimeError("No audio found. Put clips in data/speech/calm and data/speech/stress")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

if __name__ == "__main__":
    X, y = load_data()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(probability=True, kernel="rbf", C=2.0, gamma="scale", class_weight="balanced"))
    ])
    clf.fit(Xtr, ytr)
    print("Train acc:", clf.score(Xtr, ytr))
    print("Test  acc:", clf.score(Xte, yte))
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "wb") as f: pickle.dump(clf, f)
    print("Saved:", OUT_PATH)
