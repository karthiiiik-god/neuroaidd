def explain_and_recommend(audio_score, face_score, emotion, deltas=None):
    """
    Returns (why_text, tips_list, emoji)
    - deltas: dict like {"rms": +0.2, "tempo": -0.1, "EAR": -0.05, "MAR": +0.08}
    """
    reasons = []
    if emotion == "angry":   reasons.append("speech energy & pace are elevated")
    if emotion == "sad":     reasons.append("speech energy is low and rhythm is slow")
    if emotion == "stressed":reasons.append("vocal tension indicators are high")
    if face_score and face_score > 0.55: reasons.append("micro-expressions show eye narrowing / mouth tension")
    if audio_score and audio_score > 0.55:reasons.append("audio stress markers are high")

    why = " • ".join(reasons) or "signals look close to your usual baseline"

    tips = []
    if emotion in ("stressed", "angry"):
        tips += ["Try 4-7-8 breathing for 2 minutes", "Stand up, roll shoulders, 10 slow head turns",
                 "Short 5-minute walk without phone"]
    if emotion == "sad":
        tips += ["Sunlight for 10 min", "Text a friend one good thing", "Music: uplifting playlist for 5 min"]
    if not tips:
        tips = ["1 min box breathing (4-4-4-4)", "Drink water", "Write 3 lines of what’s on your mind"]

    emoji = "😖" if (audio_score or 0) > 0.65 or (face_score or 0) > 0.65 else ("😐" if (audio_score or 0)+(face_score or 0) > 0.8 else "🌞")
    return why, tips[:3], emoji
