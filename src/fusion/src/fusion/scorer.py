def fuse(audio_score, face_score, w_audio=0.6):
    if audio_score is None and face_score is None:
        return None
    if audio_score is None:
        return float(face_score)
    if face_score is None:
        return float(audio_score)
    return float(w_audio*audio_score + (1.0 - w_audio)*face_score)
