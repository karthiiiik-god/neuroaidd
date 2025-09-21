# app.py — NeuroAid (Dopamine Pack)
# - Mood Dial + Emoji
# - Streaks & Badges
# - Heatmap Calendar
# - Before vs After (confetti)
# - Weekly PDF Report
# - Baseline + Explainability light
# ---------------------------------------------------

import os, json, time, sqlite3, pickle, io, math, datetime as dt
import numpy as np
import streamlit as st

# Optional/Heavy deps loaded safely
try:
    import sounddevice as sd
    import librosa
except Exception:
    sd = None; librosa = None

try:
    import cv2
    import mediapipe as mp
except Exception:
    cv2 = None; mp = None

try:
    import pandas as pd
except Exception:
    pd = None

try:
    import plotly.graph_objects as go
except Exception:
    go = None

# PDF
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    from reportlab.lib.units import cm
    from PIL import Image
    import matplotlib.pyplot as plt
except Exception:
    canvas = None
    Image = None
    plt = None

# ---------- Page setup ----------
st.set_page_config(page_title="NeuroAid", page_icon="🧠", layout="wide")
st.title("NeuroAid — Private Mood Coach")
st.caption("Local-only • Baseline-aware • Explains & Suggests Solutions • Dopamine Pack")

# ---------- Session defaults ----------
for k, v in {
    "audio_score": None,
    "face_score": None,
    "audio_feats": None,
    "face_feats": None,
    "last_fused_before": None,
    "last_fused_after": None,
    "last_condition": None,
}.items():
    st.session_state.setdefault(k, v)

# ---------- Models (optional) ----------
SPEECH_MODEL = "src/models/speech_model.pkl"
FACE_MODEL   = "src/models/face_model.pkl"
speech_clf = None
face_clf = None
if os.path.exists(SPEECH_MODEL):
    try:
        with open(SPEECH_MODEL, "rb") as f: speech_clf = pickle.load(f)
    except Exception as e:
        st.warning(f"Speech model load issue: {e}")
if os.path.exists(FACE_MODEL):
    try:
        with open(FACE_MODEL, "rb") as f: face_clf = pickle.load(f)
    except Exception as e:
        st.warning(f"Face model load issue: {e}")

# ---------- Consent ----------
with st.expander("Consent & Privacy", expanded=True):
    agree = st.checkbox("I consent to local mic/cam processing for screening (not diagnosis).")
    st.caption("Data is stored locally. You can export/delete it. This is NOT medical advice.")

# ---------- DB ----------
DB_PATH = "neuroaid.db"
def init_db():
    with sqlite3.connect(DB_PATH) as c:
        c.execute("""CREATE TABLE IF NOT EXISTS moods(
            ts TEXT, audio REAL, face REAL, fused REAL, emoji TEXT, note TEXT
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS interventions(
            ts TEXT, before REAL, after REAL, delta REAL, label TEXT
        )""")
init_db()

def save_mood(audio, face, fused, emoji, note=""):
    with sqlite3.connect(DB_PATH) as c:
        c.execute("INSERT INTO moods VALUES(datetime('now'),?,?,?,?,?)",
                  (audio, face, fused, emoji, note))

def save_intervention(before, after, delta, label):
    with sqlite3.connect(DB_PATH) as c:
        c.execute("INSERT INTO interventions VALUES(datetime('now'),?,?,?,?)",
                  (before, after, delta, label))

def load_moods(limit=365):
    if pd is None: return None
    with sqlite3.connect(DB_PATH) as c:
        df = pd.read_sql_query(
            "SELECT * FROM moods ORDER BY ts DESC LIMIT ?",
            c, params=(limit,)
        )
    if df.empty: return df
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts")
    return df

def load_interventions(limit=100):
    if pd is None: return None
    with sqlite3.connect(DB_PATH) as c:
        df = pd.read_sql_query(
            "SELECT * FROM interventions ORDER BY ts DESC LIMIT ?",
            c, params=(limit,)
        )
    if df.empty: return df
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts")
    return df

# ---------- Baseline store ----------
BASELINE_PATH = "baseline.json"
def _empty_stats(): return {"count":0,"mean":{},"M2":{}}

def load_baseline():
    if os.path.exists(BASELINE_PATH):
        try: return json.load(open(BASELINE_PATH, "r", encoding="utf-8"))
        except: pass
    return {"audio": _empty_stats(), "face": _empty_stats()}

def save_baseline(baseline): json.dump(baseline, open(BASELINE_PATH,"w",encoding="utf-8"), indent=2)

def update_stats(stats, feats:dict):
    c=stats["count"]; mean=stats["mean"]; M2=stats["M2"]
    for k,x in feats.items():
        m=mean.get(k,0.0); m2=M2.get(k,0.0)
        c_new=c+1; delta=x-m
        m_new=m+delta/c_new; M2_new=m2+delta*(x-m_new)
        mean[k]=float(m_new); M2[k]=float(M2_new)
    stats["count"]=c+1; stats["mean"]=mean; stats["M2"]=M2; return stats

def zscores(stats, feats:dict):
    if stats["count"]<2: return None
    out={}
    for k,x in feats.items():
        if k not in stats["mean"] or k not in stats["M2"]: continue
        var=stats["M2"][k]/max(stats["count"]-1,1)
        std=float(np.sqrt(max(var,1e-12)))
        out[k]=float((x-stats["mean"][k])/std)
    return out if out else None

# ---------- Feature extraction (light & interpretable) ----------
def audio_easy(y, sr=16000):
    if librosa is None or y is None or len(y)==0: return None
    y,_ = librosa.effects.trim(y, top_db=25)
    if y.size==0: return None
    zcr=float(librosa.feature.zero_crossing_rate(y).mean())
    rms=float(librosa.feature.rms(y=y).mean())
    tempo,_ = librosa.beat.beat_track(y=y, sr=sr)
    return {"rms_mean":rms,"zcr_mean":zcr,"tempo":float(tempo)}

def face_easy_frame(frame):
    if mp is None or cv2 is None: return None
    mp_face=mp.solutions.face_mesh
    with mp_face.FaceMesh(static_image_mode=True,max_num_faces=1,refine_landmarks=True) as fm:
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        res=fm.process(rgb)
        if not res.multi_face_landmarks: return None
        lm=res.multi_face_landmarks[0]; h,w=frame.shape[:2]
        pts=np.array([[p.x*w,p.y*h] for p in lm.landmark], dtype=np.float32)
        left_eye=pts[[33,160,158,133,153,144]]
        mouth=pts[[78,81,13,311,308,402]]
        ear=(np.linalg.norm(left_eye[1]-left_eye[5])+np.linalg.norm(left_eye[2]-left_eye[4]))/(2*np.linalg.norm(left_eye[0]-left_eye[3])+1e-6)
        mar=(np.linalg.norm(mouth[1]-mouth[5])+np.linalg.norm(mouth[2]-mouth[4]))/(2*np.linalg.norm(mouth[0]-mouth[3])+1e-6)
        brow_y=float(pts[70,1]-pts[107,1]) if pts.shape[0]>108 else 0.0
        return {"ear":float(ear),"mar":float(mar),"brow_y":float(brow_y)}

# ---------- Scoring / Fusion ----------
def heuristic_audio_score(feats):
    if feats is None: return None
    # Higher energy -> more stress, lower tempo -> more stress, higher ZCR -> more stress
    base = 0.5*min(1.0, feats["rms_mean"]*8) + 0.3*min(1.0, max(0.0, 1.0-(feats["tempo"]/180.0))) + 0.2*min(1.0, feats["zcr_mean"]*10)
    return float(max(0.0, min(1.0, base)))

def heuristic_face_score(feats):
    if feats is None: return None
    ear, mar, brow = feats["ear"], feats["mar"], feats["brow_y"]
    raw = 0.6*max(0.0, 0.3-ear) + 0.3*max(0.0, mar-0.2) + 0.1*max(0.0, -brow/50.0)
    return float(max(0.0, min(1.0, raw)))

def fuse(a, f, w=0.6):
    if a is None and f is None: return None
    if a is None: return float(f)
    if f is None: return float(a)
    return float(w*a + (1.0-w)*f)

def emoji_for_score(fused):
    if fused is None: return "❔"
    if fused < 0.33: return "🌞"
    if fused < 0.66: return "😐"
    return "😖"

# ---------- Condition & Solutions ----------
def interpret_condition(zA, zF):
    # simple rules from z-scores vs baseline
    tags=[]
    if zA:
        if zA.get("rms_mean",0)>1 or zA.get("zcr_mean",0)>1: tags.append("stress")
        if zA.get("tempo",0)<-1 or zA.get("rms_mean",0)<-1: tags.append("low_mood")
    if zF:
        if zF.get("ear",0)<-1 or zF.get("mar",0)>1 or zF.get("brow_y",0)<-1: tags.append("stress")
    if not tags: return "normal"
    return tags[0]

SOLUTIONS = {
    "stress": [
        "Try 2 minutes of 4–7–8 breathing",
        "Stand up and stretch your shoulders",
        "Listen to calming instrumental music"
    ],
    "low_mood": [
        "Write down one thing you’re grateful for today",
        "Take a 5-minute walk outside",
        "Call a friend or family member"
    ],
    "voice_fatigue": [
        "Sip warm water slowly",
        "Do a soft humming exercise",
        "Take a short voice rest"
    ]
}

# ---------- UI: Capture ----------
left, right = st.columns(2)

with left:
    st.subheader("🎙️ Audio Check")
    if not agree:
        st.warning("Consent needed for microphone.")
    else:
        if sd is None or librosa is None:
            st.error("Missing audio libs. Install: pip install sounddevice librosa")
        else:
            dur = st.slider("Record length (seconds)", 5, 20, 8)
            if st.button("Record Audio"):
                try:
                    y = sd.rec(int(16000*dur), samplerate=16000, channels=1, dtype='float32'); sd.wait()
                    y = y.squeeze()
                    A = audio_easy(y, 16000)
                    st.session_state.audio_feats = A
                    st.session_state.audio_score = heuristic_audio_score(A)
                    st.success(f"Audio stress score: {st.session_state.audio_score:.2f}")
                except Exception as e:
                    st.error(f"Audio error: {e}")

with right:
    st.subheader("📷 Face Check")
    if not agree:
        st.warning("Consent needed for webcam.")
    else:
        if cv2 is None or mp is None:
            st.error("Missing video libs. Install: pip install opencv-python mediapipe")
        else:
            if st.button("Capture Face"):
                try:
                    cap = cv2.VideoCapture(0)
                    ok, frame = cap.read()
                    cap.release()
                    if not ok:
                        st.error("Could not access camera.")
                    else:
                        F = face_easy_frame(frame)
                        st.session_state.face_feats = F
                        st.session_state.face_score = heuristic_face_score(F)
                        st.success(f"Face stress score: {st.session_state.face_score:.2f}")
                except Exception as e:
                    st.error(f"Camera error: {e}")

# ---------- Fused score & Mood Dial ----------
a = st.session_state.audio_score
f = st.session_state.face_score
fused = fuse(a, f)
st.session_state.last_fused_after = fused  # current

def mood_dial(value_0to1):
    if go is None:
        st.write("Install plotly for dial: pip install plotly")
        return
    val = 100.0*(1.0 - (value_0to1 or 0.0))  # higher is calmer
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=val,
        title={"text":"Mood Index (higher=calmer)"},
        delta={'reference': 50},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'thickness': 0.25},
               'steps': [
                   {'range': [0, 33], 'color': "#ff6666"},
                   {'range': [33, 66], 'color': "#ffd166"},
                   {'range': [66, 100], 'color': "#7bd88f"},
               ]},
        number={'suffix': "/100"}
    ))
    fig.update_layout(height=280, margin=dict(l=20,r=20,t=40,b=20))
    st.plotly_chart(fig, use_container_width=True)

mood_dial(fused)
st.markdown(f"### Mood: {emoji_for_score(fused)}")

# ---------- Baseline & Explainability ----------
st.subheader("Baseline & Explanations")
baseline = load_baseline()
st.caption(f"Baseline samples — Audio: {baseline['audio']['count']} | Face: {baseline['face']['count']}")

colB1, colB2, colB3 = st.columns(3)
with colB1:
    if st.button("➕ Add Audio to Baseline"):
        if st.session_state.audio_feats:
            baseline["audio"] = update_stats(baseline["audio"], st.session_state.audio_feats)
            save_baseline(baseline); st.success("Audio baseline updated.")
        else:
            st.warning("Record audio first.")
with colB2:
    if st.button("➕ Add Face to Baseline"):
        if st.session_state.face_feats:
            baseline["face"] = update_stats(baseline["face"], st.session_state.face_feats)
            save_baseline(baseline); st.success("Face baseline updated.")
        else:
            st.warning("Capture face first.")
with colB3:
    st.write("")

zA = zscores(baseline["audio"], st.session_state.audio_feats or {}) if st.session_state.audio_feats else None
zF = zscores(baseline["face"],  st.session_state.face_feats  or {}) if st.session_state.face_feats else None
cond = "normal"
if zA or zF:
    # quick chips
    chips_cols = st.columns(3)
    def chip(label, value):
        chips_cols.pop(0).metric(label, f"{value:+.2f}")
    # pick top by abs(z)
    items=[]
    if zA: items += [(k, v) for k,v in zA.items()]
    if zF: items += [(k, v) for k,v in zF.items()]
    items.sort(key=lambda kv: abs(kv[1]), reverse=True)
    for (k,v) in items[:3]:
        name = {"rms_mean":"Energy (RMS)","zcr_mean":"Noisiness (ZCR)","tempo":"Tempo",
                "ear":"Eye tension (EAR)","mar":"Mouth tension (MAR)","brow_y":"Brow"} .get(k,k)
        chip(name, v)
    # condition + solutions
    cond = ("voice_fatigue" if (zA and zA.get("rms_mean",0)<-1 and (zA.get("tempo",0)<-1)) else interpret_condition(zA,zF))

if cond != "normal":
    st.markdown(f"**Detected:** `{cond}`")
    if cond == "stress":
        st.write("Signals indicate tension — higher energy/noise or narrowed eyes/tense mouth vs your baseline.")
    elif cond == "low_mood":
        st.write("Signals show lower tempo and energy than your baseline, which may reflect fatigue or low mood.")
    elif cond == "voice_fatigue":
        st.write("Vocal energy and tempo are down, suggesting tiredness or voice strain.")
    st.markdown("**Suggested Actions:**")
    for s in SOLUTIONS.get(cond, []):
        st.write("✔️ ", s)

# ---------- Save + Before/After + Confetti ----------
note = st.text_input("Note (optional)")
emoji = emoji_for_score(fused)

cols_save = st.columns(3)
with cols_save[0]:
    if st.button("💾 Save Today"):
        try:
            save_mood(float(a or 0.0), float(f or 0.0), float(fused or 0.0), emoji, note)
            st.success("Saved ✅")
        except Exception as e:
            st.error(f"Save failed: {e}")

with cols_save[1]:
    if st.button("Set as BEFORE"):
        st.session_state.last_fused_before = fused
        st.info(f"Baseline set: before={fused:.2f} (lower later is better)")

with cols_save[2]:
    if st.button("Re-check & Compare"):
        before = st.session_state.last_fused_before
        after  = fused
        if before is None or after is None:
            st.warning("Capture scores first and set BEFORE.")
        else:
            delta = (after - before)
            # lower is better → negative delta is good
            msg = f"Change (after - before): {delta:+.2f}"
            if delta < -0.05:
                st.success(msg + " 🎉 Improvement!")
                st.balloons()
            elif delta > 0.05:
                st.warning(msg + " (a bit higher stress)")
            else:
                st.info(msg + " (about the same)")
            try:
                save_intervention(float(before), float(after), float(delta), cond)
            except Exception: pass

# ---------- Streaks, Badges, Heatmap ----------
st.subheader("Progress & Motivation")

def compute_streaks(df):
    if df is None or df.empty: return 0, 0
    days = df["ts"].dt.date.unique().tolist()
    days.sort()
    # current streak
    today = dt.date.today()
    streak = 0
    cur = today
    while cur in days:
        streak += 1
        cur = cur - dt.timedelta(days=1)
    # max streak
    max_streak = 0
    count = 1
    for i in range(1, len(days)):
        if (days[i] - days[i-1]).days == 1:
            count += 1
        else:
            max_streak = max(max_streak, count)
            count = 1
    max_streak = max(max_streak, count)
    return streak, max_streak

def award_badges(df):
    badges=[]
    if df is None or df.empty: return badges
    streak, max_streak = compute_streaks(df)
    if streak >= 3: badges.append("🔥 3-day Streak")
    if streak >= 7: badges.append("🏆 7-day Streak")
    # Calm Week badge: average fused < 0.33 over last 7 days
    last7 = df[df["ts"] >= (pd.Timestamp.today() - pd.Timedelta(days=7))]
    if not last7.empty and last7["fused"].mean() < 0.33:
        badges.append("🧘 Calm Week")
    # Consistency: 5+ logs in last 7 days
    if not last7.empty and len(last7) >= 5:
        badges.append("📈 Consistency King")
    return badges

df = load_moods(365)
if pd is None:
    st.info("Install pandas to see progress widgets.")
else:
    # Streak counters
    cs, ms = compute_streaks(df)
    c1, c2 = st.columns(2)
    c1.metric("Current Streak", f"{cs} days")
    c2.metric("Max Streak", f"{ms} days")

    # Badges
    badges = award_badges(df)
    if badges:
        st.markdown("**Badges:** " + " | ".join(badges))
    else:
        st.caption("No badges yet—log daily to unlock rewards.")

    # Heatmap calendar (week x weekday)
    if not df.empty and plt is not None:
        # Build daily avg fused
        daily = df.groupby(df["ts"].dt.date)["fused"].mean()
        # Map recent 8 weeks
        today = dt.date.today()
        start = today - dt.timedelta(days=7*8-1)
        dates = [start + dt.timedelta(days=i) for i in range(7*8)]
        values = [daily.get(d, np.nan) for d in dates]
        # Heatmap grid: rows = Mon..Sun, cols = weeks
        grid = np.full((7, 8), np.nan)
        for idx, d in enumerate(dates):
            week = idx // 7
            dow = (d.weekday())  # 0=Mon..6=Sun
            grid[dow, week] = values[idx]

        fig, ax = plt.subplots(figsize=(8, 2.8))
        im = ax.imshow(grid, aspect='auto', vmin=0, vmax=1, cmap="RdYlGn_r")
        ax.set_yticks(range(7)); ax.set_yticklabels(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
        ax.set_xticks(range(8)); ax.set_xticklabels([f"W{-7+i}" for i in range(8)])
        ax.set_title("Mood Heatmap (8 weeks, red=stress, green=calm)")
        plt.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
        st.pyplot(fig)

# ---------- Weekly PDF Report ----------
st.subheader("Weekly PDF Report")
def generate_weekly_pdf(path="neuroaid_weekly_report.pdf"):
    if canvas is None or pd is None: 
        st.error("Missing libs for PDF (reportlab) or pandas.")
        return None

    df = load_moods(60)
    if df is None or df.empty:
        st.warning("No data to include.")
        return None

    # last 7 days
    last7 = df[df["ts"] >= (pd.Timestamp.today() - pd.Timedelta(days=7))]
    avg = last7["fused"].mean() if not last7.empty else None

    # build a small line chart image
    chart_path = "weekly_chart.png"
    try:
        if plt is not None and not last7.empty:
            fig, ax = plt.subplots(figsize=(5,2))
            ax.plot(last7["ts"], 1.0 - last7["fused"], linewidth=2)
            ax.set_title("Calm Index (higher=calmer)")
            ax.set_ylim(0,1)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(chart_path, dpi=140)
            plt.close(fig)
    except Exception:
        pass

    # Generate PDF
    c = canvas.Canvas(path, pagesize=A4)
    W, H = A4
    y = H - 2*cm
    c.setFont("Helvetica-Bold", 16)
    c.drawString(2*cm, y, "NeuroAid — Weekly Report")
    y -= 1.2*cm
    c.setFont("Helvetica", 11)
    c.drawString(2*cm, y, f"Date: {dt.date.today().isoformat()}")
    y -= 0.8*cm

    if avg is not None:
        calm_idx = int((1.0 - avg)*100)
        c.drawString(2*cm, y, f"Average Calm Index: {calm_idx}/100")
        y -= 0.6*cm

    # top notes this week (first 3)
    notes = last7.dropna(subset=["note"])["note"].tolist() if not last7.empty else []
    if notes:
        c.drawString(2*cm, y, "Notes (first 3):")
        y -= 0.5*cm
        for n in notes[:3]:
            c.drawString(2.2*cm, y, f"- {n[:90]}")
            y -= 0.45*cm

    # insert chart if present
    if os.path.exists(chart_path):
        try:
            img = Image.open(chart_path)
            iw, ih = img.size
            maxw, maxh = 16*cm, 6*cm
            scale = min(maxw/iw, maxh/ih)
            c.drawImage(ImageReader(img), 2*cm, y - (ih*scale), width=iw*scale, height=ih*scale)
            y -= (ih*scale + 0.8*cm)
        except Exception:
            pass

    c.setFont("Helvetica-Oblique", 9)
    c.drawString(2*cm, 2*cm, "Private & offline. Screening only — not medical advice.")
    c.showPage(); c.save()
    return path

colPDF1, colPDF2 = st.columns(2)
with colPDF1:
    if st.button("Generate Weekly PDF"):
        pdf_path = generate_weekly_pdf()
        if pdf_path and os.path.exists(pdf_path):
            st.success("Report generated.")
            with open(pdf_path, "rb") as f:
                st.download_button("Download Weekly Report PDF", f, file_name=os.path.basename(pdf_path), mime="application/pdf")

with colPDF2:
    # simple export of raw data
    if st.button("Export Raw Data (JSON)"):
        try:
            df = load_moods(365)
            data = df.to_dict(orient="records") if df is not None else []
            with open("my_neuroaid_export.json", "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            st.success("Exported → my_neuroaid_export.json")
            with open("my_neuroaid_export.json","rb") as f:
                st.download_button("Download JSON", f, file_name="my_neuroaid_export.json", mime="application/json")
        except Exception as e:
            st.error(f"Export failed: {e}")

# ---------- Footer ----------
st.info("⚠️ NeuroAid is a learning/research tool. It does not diagnose conditions.")
