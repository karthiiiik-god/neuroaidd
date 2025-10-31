# app.py — NeuroAid (Pushpa Black + Neon theme)
import os, json, time, pickle
import numpy as np
import streamlit as st

# --- optional heavy deps (guarded) ---
try:
    import sounddevice as sd
    import librosa
except Exception:
    sd = None; librosa = None

try:
    import cv2
except Exception:
    cv2 = None

try:
    import pandas as pd
except Exception:
    pd = None

# --- your modules ---
from src.auth.auth import login_block
from src.ai.suggester import explain_and_recommend
from src.ai.counselor import reply as counselor_reply
from src.audio.emotion import predict_emotion
from src.video.realtime import capture_face_live
from src.db.store import init_db, migrate_db, save_mood as db_save_mood, export_all

# --- optional model ---
SPEECH_MODEL = "src/models/speech_model.pkl"
speech_clf = None
if os.path.exists(SPEECH_MODEL):
    try:
        with open(SPEECH_MODEL, "rb") as f:
            speech_clf = pickle.load(f)
    except Exception:
        speech_clf = None

# ---------- THEME ----------
ACCENT_NEON   = "#00FFA3"   # neon green
ACCENT_RUBY   = "#FF375F"   # ruby red
ACCENT_GOLD   = "#F4C430"   # golden
SURFACE_A     = "rgba(255,255,255,.03)"
SURFACE_B     = "rgba(255,255,255,.02)"
BORDER        = "rgba(255,255,255,.08)"
BORDER_SOFT   = "rgba(255,255,255,.06)"

st.set_page_config(page_title="NeuroAid", page_icon="🧠", layout="wide")
st.markdown(
    f"""
    <style>
      /* page chrome */
      .block-container{{padding-top:1.2rem;}}
      .stTabs [data-baseweb="tab-list"] button {{ gap:.5rem }}
      .stTabs [data-baseweb="tab-highlight"]{{ background:{ACCENT_NEON}22 }}

      /* common atoms */
      .pill {{
        display:inline-flex;align-items:center;gap:.5rem;
        padding:.45rem .8rem;border-radius:999px;
        background:{SURFACE_A};border:1px solid {BORDER};
        font-size:.9rem
      }}
      .btn-ghost {{
        border:1px solid {BORDER}!important;
        background: {SURFACE_A}!important;
        box-shadow: 0 0 0 0 rgba(0,0,0,0);
      }}
      .btn-ghost:hover{{ box-shadow:0 0 0 2px {ACCENT_NEON}33 inset }}

      .card {{
        background:linear-gradient(180deg, {SURFACE_A}, {SURFACE_B});
        border:1px solid {BORDER};
        padding:14px;border-radius:14px;
        box-shadow: 0 10px 30px rgba(0,0,0,.25), 0 0 30px {ACCENT_NEON}0A;
      }}

      /* Fusion */
      .fusion-wrap{{
        display:flex; gap:18px; align-items:center;
        padding:16px; border-radius:16px;
        background:linear-gradient(180deg, {SURFACE_A}, {SURFACE_B});
        border:1px solid {BORDER_SOFT};
        box-shadow: 0 0 24px {ACCENT_NEON}12 inset, 0 0 8px {ACCENT_NEON}22;
      }}
      .fusion-meta .k{{ font-size:12px; opacity:.8; margin-bottom:4px; }}
      .fusion-meta .v{{ font-size:32px; font-weight:900; margin-bottom:10px;
                        background: linear-gradient(90deg,{ACCENT_NEON},#7CF4C2);
                        -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
      .badge{{
        display:inline-flex; align-items:center; gap:8px;
        padding:6px 10px; border-radius:999px; font-weight:700; color:#0A0F0D;
        background:{ACCENT_NEON};
        box-shadow:0 0 18px {ACCENT_NEON}55, inset 0 0 10px #fff3;
      }}
      .badge.ruby{{ background:{ACCENT_RUBY}; color:white; box-shadow:0 0 18px {ACCENT_RUBY}66 }}
      .badge.gold{{ background:{ACCENT_GOLD}; color:#111; box-shadow:0 0 18px {ACCENT_GOLD}66 }}

      .rows{{ display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:12px;}}
      .row{{
        background:{SURFACE_A}; border-radius:10px; padding:8px 10px;
        border:1px solid {BORDER_SOFT};
      }}
      .bar{{ height:6px; border-radius:999px; background:{BORDER_SOFT}; overflow:hidden; margin-top:6px;}}
      .bar > span{{ display:block; height:100%; background:linear-gradient(90deg,{ACCENT_NEON},#16a34a); }}

      .tip{{opacity:.8}}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("NeuroAid — Private, Explainable Mood Coach")

# ---------- DB & Auth ----------
init_db(); migrate_db()
st.session_state.setdefault("user", None)
if not st.session_state.user:
    login_block()
    st.stop()
USER = st.session_state.user
st.caption(f"Signed in as **{USER['username']}** — fully local & private. Export/Delete anytime.")

# ---------- Consent ----------
with st.expander("Consent & Privacy", expanded=True):
    agree = st.checkbox("I consent to local mic/cam processing for screening (not diagnosis).")
    st.caption("This is not a medical device. Data remains on this device unless you export it.")

# ---------- helpers ----------
def fuse(a, f, w=0.6):
    if a is None and f is None: return None
    if a is None: return float(f)
    if f is None: return float(a)
    return float(w*a + (1-w)*f)

def safe_record_audio(seconds=8, sr=16000):
    if not agree or sd is None: return (None, None)
    try:
        dev = sd.default.device
        if dev is None or (isinstance(dev, (list, tuple)) and dev[0] is None):
            idx = None
            for i, d in enumerate(sd.query_devices()):
                if int(d.get("max_input_channels", 0)) > 0:
                    idx = i; break
            if idx is None:
                st.error("No microphone input device found.")
                return None, None
            sd.default.device = (idx, sd.default.device[1] if isinstance(sd.default.device, (list, tuple)) else None)
        frames = int(seconds * sr)
        y = sd.rec(frames, samplerate=sr, channels=1, dtype="float32")
        sd.wait()
        return np.asarray(y, dtype=np.float32).reshape(-1), sr
    except Exception as e:
        st.error(f"Microphone error: {e}")
        return None, None

def try_capture_face(secs: int, device_index: int):
    if not agree or cv2 is None: return None
    try:
        return capture_face_live(seconds=secs, device_index=int(device_index))
    except TypeError:
        try:
            frames = max(10, int(secs * 15))
            return capture_face_live(frames=frames, device_index=int(device_index))
        except Exception as e:
            st.error(f"Camera error: {e}")
            return None
    except Exception as e:
        st.error(f"Camera error: {e}")
        return None

def mood_emoji_and_badge(name: str):
    name = (name or "neutral").lower()
    if name in ("stressed","angry"):     return "🔥", "ruby"
    if name in ("sad",):                 return "😔", "gold"   # gold to highlight support
    if name in ("happy","calm","relaxed"): return "😊", ""     # default neon
    return "😐", ""  # neutral

def radial_gauge_svg(value: int, size: int = 140) -> str:
    v = max(0, min(100, int(value)))
    R = 56
    C = R * 2 * 3.14159265
    dash = C * v / 100
    gap  = C - dash
    # gradient tuned to neon -> gold -> ruby
    return f"""
    <svg width="{size}" height="{size}" viewBox="0 0 140 140" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%"  stop-color="{ACCENT_NEON}"/>
          <stop offset="55%" stop-color="{ACCENT_GOLD}"/>
          <stop offset="100%" stop-color="{ACCENT_RUBY}"/>
        </linearGradient>
      </defs>
      <circle cx="70" cy="70" r="{R}" fill="none" stroke="rgba(255,255,255,.10)" stroke-width="14"/>
      <circle cx="70" cy="70" r="{R}" fill="none" stroke="url(#g)" stroke-width="14"
              stroke-linecap="round" transform="rotate(-90 70 70)"
              stroke-dasharray="{dash} {gap}"/>
      <filter id="glow"><feGaussianBlur stdDeviation="2.5" result="coloredBlur"/><feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
      <circle cx="70" cy="70" r="{R}" fill="none" stroke="url(#g)" stroke-width="2" opacity=".35" filter="url(#glow)"/>
      <text x="70" y="74" font-size="28" text-anchor="middle" fill="#fff" font-weight="800">{v}</text>
      <text x="70" y="95" font-size="10" text-anchor="middle" fill="rgba(255,255,255,.7)">/100</text>
    </svg>
    """

def fusion_styles():  # already injected above, keep hook for readability
    pass

# ---------- TABS ----------
tab_check, tab_coach, tab_chat, tab_progress, tab_settings = st.tabs(
    ["✅ Check", "🧠 Coach", "💬 Chat", "📊 Progress", "⚙️ Settings"]
)

# ---------- CHECK ----------
with tab_check:
    colA, colB, colC = st.columns([1,1,1])
    audio_score = None; face_score = None; emotion_label = "neutral"; emo_prob = 0.5

    # --- Audio ---
    with colA:
        st.subheader("Audio")
        secs = st.slider("Record length (seconds)", 4, 15, 6, key="rec_secs")
        if st.button("🎙️ Record", use_container_width=True):
            if not agree:
                st.info("Give consent to enable mic.")
            elif sd is None or librosa is None:
                st.error("Install audio libs: `pip install sounddevice librosa`")
            else:
                y, sr = safe_record_audio(secs, 16000)
                if y is not None:
                    rms = float(np.sqrt((y.astype(np.float32)**2).mean()))
                    audio_score = float(min(1.0, max(0.0, rms*8)))
                    try:
                        emotion_label, emo_prob = predict_emotion(y, sr, speech_clf)
                    except Exception:
                        emotion_label, emo_prob = "neutral", 0.5
                    st.success(f"Audio stress≈ {audio_score:.2f} | Emotion: {emotion_label} ({emo_prob:.2f})")

    # --- Face ---
    with colB:
        st.subheader("Face")
        secs_face = st.slider("Capture seconds", 3, 10, 5, key="cam_secs")
        cam_idx = st.number_input("Webcam index", min_value=0, max_value=5, value=0, step=1)
        if st.button("📷 Live Check", use_container_width=True):
            if not agree:
                st.info("Give consent to enable camera.")
            elif cv2 is None:
                st.error("Install video libs: `pip install opencv-python`")
            else:
                feat = try_capture_face(int(secs_face), int(cam_idx))
                if feat is None:
                    st.warning("Face not detected. Try better lighting and center your face.")
                else:
                    try:
                        ear, mar, brow = np.asarray(feat, dtype=float).tolist()
                    except Exception:
                        st.warning("Unexpected face features; using fallback.")
                        ear, mar, brow = 0.3, 0.2, 0.0
                    raw = 0.6*max(0.0, 0.3-ear) + 0.4*max(0.0, mar-0.2) + 0.1*max(0.0, -brow/50.0)
                    face_score = float(max(0.0, min(1.0, raw)))
                    st.success(f"Face stress≈ {face_score:.2f}")

    # --- Fusion (Neon card) ---
    with colC:
        st.subheader("Fusion")

    fused = fuse(audio_score, face_score)
    mood_idx = (1.0 - (fused or 0.0)) * 100
    mood_idx = float(np.clip(mood_idx, 0, 100))

    # color by state
    if mood_idx > 80:
        mood_color = "#00FFAA"   # calm neon green
        ring_from, ring_to = "#00FFAA", "#1EE3FF"
    elif mood_idx > 60:
        mood_color = "#FFD700"   # balanced gold
        ring_from, ring_to = "#FFD700", "#FF8C00"
    else:
        mood_color = "#FF4B4B"   # stressed red
        ring_from, ring_to = "#FF4B4B", "#FF8C8C"

    emotion_display = (emotion_label or "neutral").capitalize()
    emo_prob_display = f"{emo_prob * 100:.0f}%" if isinstance(emo_prob, (int, float)) else "–"

    # animated card + ring + pulse
    st.markdown(f"""
    <style>
      /* isolate styles to this card only */
      .fusion-card {{
        background: radial-gradient(1200px 400px at 100% -10%, rgba(30,227,255,0.08), transparent 40%) ,
                    linear-gradient(135deg, rgba(0,255,170,0.12), rgba(0,0,0,0.45));
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 22px;
        padding: 26px 24px 28px;
        box-shadow: 0 0 0 1px rgba(255,255,255,0.03), 0 12px 40px rgba(0,0,0,0.35);
      }}
      .fusion-ring {{
        width: 164px; height: 164px; border-radius: 50%;
        margin: 8px auto 2px; position: relative;
        background: conic-gradient({ring_from} 0deg, {ring_to} 120deg, rgba(255,255,255,0.08) 120deg);
        filter: drop-shadow(0 0 18px rgba(0,0,0,0.35));
        animation: spin-slow 18s linear infinite;
      }}
      .fusion-ring::before {{
        content: "";
        position: absolute; inset: 12px; border-radius: 50%;
        background: #0f1116; box-shadow: inset 0 0 0 1px rgba(255,255,255,0.06);
      }}
      .fusion-num {{
        position: absolute; inset: 0; display:flex; align-items:center; justify-content:center;
        color: {mood_color}; font-weight: 900; font-size: 44px; letter-spacing: -1px;
        text-shadow: 0 0 28px rgba(0,0,0,0.45), 0 0 12px {mood_color}55;
        animation: pulse-glow 2.4s ease-in-out infinite;
      }}
      .fusion-sub {{
        position: absolute; bottom: 20px; width:100%; text-align:center;
        font-size: 11px; color: #9aa3ad; letter-spacing: .1em;
      }}
      .badge {{
        display:inline-flex; gap:.5rem; align-items:center;
        padding: 6px 10px; border-radius: 999px;
        background: rgba(255,255,255,0.06); color:#cfd6dd; font-size:12px;
        border: 1px solid rgba(255,255,255,0.08);
      }}
      .delta {{ color: #00FFAA; }}
      @keyframes pulse-glow {{
        0%, 100% {{ transform: scale(1);     text-shadow: 0 0 12px {mood_color}44, 0 0 30px rgba(0,0,0,0.35); }}
        50%      {{ transform: scale(1.06);  text-shadow: 0 0 18px {mood_color}88, 0 0 40px rgba(0,0,0,0.45); }}
      }}
      @keyframes spin-slow {{ from {{ transform: rotate(0deg); }} to {{ transform: rotate(360deg); }} }}
    </style>

    <div class="fusion-card">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
        <h3 style="margin:0; color:#dfe6ee; font-size:20px;">Mood Index</h3>
        <span class="badge">
          <span style="opacity:.7;">Emotion</span>
          <strong style="color:#fff;">{emotion_display}</strong>
          <span class="delta">↑ {emo_prob_display}</span>
        </span>
      </div>

      <div class="fusion-ring">
        <div class="fusion-num">{mood_idx:.0f}</div>
        <div class="fusion-sub">/100</div>
      </div>

      <div style="margin-top:18px; display:grid; grid-template-columns: 1fr 1fr; gap:10px;">
        <div style="background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.06); border-radius:12px; padding:10px 12px;">
          <div style="font-size:12px; color:#9aa3ad; margin-bottom:6px;">Audio score</div>
          <div style="height:7px; background:#1b2129; border-radius:6px; overflow:hidden;">
            <div style="height:100%; width:{(audio_score or 0)*100:.0f}%; background:linear-gradient(90deg, #00ffae, #1ee3ff);"></div>
          </div>
        </div>
        <div style="background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.06); border-radius:12px; padding:10px 12px;">
          <div style="font-size:12px; color:#9aa3ad; margin-bottom:6px;">Face score</div>
          <div style="height:7px; background:#1b2129; border-radius:6px; overflow:hidden;">
            <div style="height:100%; width:{(face_score or 0)*100:.0f}%; background:linear-gradient(90deg, #ffd700, #ff8c00);"></div>
          </div>
        </div>
      </div>

      <p style="color:#98a2ad; font-size:13px; margin-top:16px;">
        Tip: Add a few ‘good’ readings to make personal comparisons meaningful.
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    note = st.text_input("Note (optional)")
    if st.button("💾 Save Today", use_container_width=True):
        why, tips, emoji = explain_and_recommend(audio_score, face_score, emotion_label)
        db_save_mood(USER['id'], audio_score or 0.0, face_score or 0.0, fused or 0.0, emoji, note, emotion_label, tips)
        st.success(f"Saved with emoji {emoji} and emotion {emotion_label}.")

# ---------- COACH ----------
with tab_coach:
    st.subheader("AI Coach — Why & What to do")
    a = st.number_input("Audio score (0..1)", 0.0, 1.0, 0.5, 0.01, key="coach_a")
    f = st.number_input("Face score (0..1)", 0.0, 1.0, 0.5, 0.01, key="coach_f")
    emo = st.selectbox("Emotion", ["neutral","stressed","sad","angry","happy"], index=0)
    why, tips, emoji = explain_and_recommend(a, f, emo)
    st.write(f"**Why**: {why}")
    st.write("**Top tips now:**")
    for t in tips: st.write(f"• {t}")
    st.write(f"**Mood emoji:** {emoji}")

# ---------- CHAT ----------
with tab_chat:
    st.subheader("Counselor Chat (offline)")
    chat = st.session_state.get("chat", [])
    for role, msg in chat:
        st.chat_message(role).write(msg)
    prompt = st.chat_input("Type how you feel…")
    if prompt:
        st.session_state.chat = chat + [("user", prompt)]
        st.chat_message("user").write(prompt)
        bot = counselor_reply(prompt)
        st.session_state.chat.append(("assistant", bot))
        st.chat_message("assistant").write(bot)

# ---------- PROGRESS ----------
with tab_progress:
    st.subheader("Your Progress")
    if pd is None:
        st.info("Install pandas to view charts.")
    else:
        rows = export_all(USER['id'])
        if not rows:
            st.caption("No entries yet.")
        else:
            cols = ["ts","user_id","audio","face","fused","emoji","note","emotion","tips_used"]
            df = pd.DataFrame(rows, columns=cols).sort_values("ts")
            df["ts"] = pd.to_datetime(df["ts"])
            st.line_chart(df.set_index("ts")[["audio","face","fused"]], height=260)
            st.dataframe(df[["ts","emoji","emotion","note"]].tail(10), use_container_width=True)

# ---------- SETTINGS ----------
with tab_settings:
    st.subheader("Settings")
    st.write("• To reset database, delete `neuroaid.db` file (app will recreate).")
    if st.button("Sign out"):
        st.session_state.user = None
        st.experimental_rerun()
