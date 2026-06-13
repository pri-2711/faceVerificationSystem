# =============================================================
#  app.py  —  Streamlit UI  (entry point)
# =============================================================

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import base64
from datetime import datetime
from PIL import Image

from database   import (
    load_all_users, save_user, delete_user,
    get_recent_logs, count_users,
    save_log, save_unknown, count_unknowns_in_window
)
from embedding  import get_single_embedding, get_all_embeddings
from similarity import normalize, find_best_match
from verify     import verify_image
from register   import register_bulk, register_from_webcam
from alert      import alert_unknown_face, alert_multiple_unknowns, alert_duplicate_entry
import config


# ── Folder picker ─────────────────────────────────────────────
def pick_folder():
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.wm_attributes("-topmost", True)
    folder = filedialog.askdirectory(title="Select dataset folder")
    root.destroy()
    return folder


# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Face Verification",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Styles ────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&family=DM+Mono:wght@400;500&display=swap');

*, html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }

[data-testid="stSidebar"] {
    background: #0f0f0f !important;
    border-right: 1px solid #1c1c1c;
}
[data-testid="stSidebar"] * { color: #c8c8c8 !important; }
[data-testid="stSidebar"] .stRadio label {
    font-size: 13px !important; padding: 8px 12px !important;
    border-radius: 6px !important; display: block;
}
[data-testid="stSidebar"] .stRadio label:hover { background: #1a1a1a !important; }

.block-container { padding: 36px 44px 40px !important; max-width: 1080px; }

/* headings — light background assumed */
.pg-title { font-size: 21px; font-weight: 600; color: #111;
            letter-spacing: -0.03em; margin-bottom: 3px; }
.pg-sub   { font-size: 13px; color: #777; font-weight: 300; margin-bottom: 28px; }

/* stat card */
.sc { background:#f9f9f9; border:1px solid #e8e8e8; border-radius:10px; padding:18px 22px; }
.sc-l { font-size:10px; font-weight:600; letter-spacing:0.1em; text-transform:uppercase;
        color:#999; margin-bottom:8px; }
.sc-v { font-size:30px; font-weight:600; color:#111; line-height:1; }

/* badges */
.badge { display:inline-block; padding:3px 10px; border-radius:20px;
         font-size:10px; font-weight:600; letter-spacing:0.05em; text-transform:uppercase; }
.b-ok   { background:#22c55e; color:white; }
.b-fail { background:#ef4444; color:white; }
.b-dup  { background:#fff3e0; color:#bf360c; }

/* result table */
.rt-wrap { border:1px solid #e8e8e8; border-radius:10px; overflow:hidden; margin-top:14px; }
.rt { width:100%; border-collapse:collapse; font-size:13px; }
.rt thead { background:#f9f9f9; }
.rt th { text-align:left; font-size:10px; font-weight:600; letter-spacing:0.09em;
         text-transform:uppercase; color:#999; padding:11px 16px;
         border-bottom:1px solid #e8e8e8; }
.rt td { padding:12px 16px; border-bottom:1px solid #f2f2f2;
         font-family:'DM Mono',monospace; font-size:12px; color:#222; }
.rt tr:last-child td { border-bottom:none; }
.rt tr:hover td { background:#fafafa; }

/* notice box */
.notice { background:#f7f7f7; border:1px solid #e4e4e4; border-radius:8px;
          padding:13px 18px; font-size:13px; color:#444; line-height:1.7;
          margin-bottom:18px; }
.notice code { background:#ececec; padding:1px 6px; border-radius:4px;
               font-family:'DM Mono',monospace; font-size:11px; color:#333; }

/* path display */
.path-box { background:#f4f4f4; border:1px solid #e0e0e0; border-radius:7px;
            padding:9px 14px; font-family:'DM Mono',monospace; font-size:12px;
            color:#333; min-height:38px; word-break:break-all; }
.path-empty { color:#bbb; }

.div-line { height:1px; background:#e8e8e8; margin:22px 0; }

/* buttons */
.stButton > button {
    background:#111 !important; color:#fff !important;
    border:none !important; border-radius:7px !important;
    font-size:13px !important; font-weight:500 !important;
    padding:9px 22px !important; letter-spacing:0.01em;
    transition: background 0.15s !important;
}
.stButton > button:hover { background:#333 !important; }

/* inputs */
.stTextInput input {
    border-radius:7px !important; border-color:#ddd !important;
    font-size:13px !important; padding:9px 12px !important; color:#111 !important;
}
label { font-size:12px !important; font-weight:500 !important;
        color:#555 !important; letter-spacing:0.02em; }
.stProgress > div > div { background:#111 !important; border-radius:4px; }
[data-testid="stFileUploader"] {
    border:1.5px dashed #ddd !important; border-radius:10px !important;
}
/* dataframe text readable */
[data-testid="stDataFrame"] { color: #111 !important; }

/* Make text white */
.block-container, .block-container * { color: white !important; }
.stTextInput input { color: white !important; background: #1a1a1a !important; }
label { color: white !important; }
.notice { background: #1a1a1a !important; border-color: #333 !important; color: white !important; }
.notice code { background: #333 !important; color: white !important; }
.sc { background: #1a1a1a !important; border-color: #333 !important; }
.sc-l { color: #ccc !important; }
.sc-l-black { color: black !important; }
.sc-v { color: white !important; }
.path-box { background: #1a1a1a !important; border-color: #333 !important; color: white !important; }
.path-empty { color: #777 !important; }
.rt thead { background: #1a1a1a !important; }
.rt th { color: #ccc !important; border-bottom-color: #333 !important; }
.rt td { color: white !important; border-bottom-color: #333 !important; }
.rt tr:hover td { background: #333 !important; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────
for key, default in [
    ("recent_entries", {}),      # { name: last_verified_datetime }
    ("last_log_time",  {}),      # { name: last_logged_datetime } — throttle logs
    ("run_cam",        False),
    ("alert_sent",     False),
    ("dataset_path",   config.DATASET_PATH),
]:
    if key not in st.session_state:
        st.session_state[key] = default

LOG_INTERVAL_SEC = 5   # minimum seconds between logs for the same person


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='padding:12px 16px 0;'>"
        "<div style='font-size:15px;font-weight:600;color:#eee;"
        "letter-spacing:-0.02em;margin-bottom:2px;'>FaceVerify</div>"
        "<div style='font-size:11px;color:#555;margin-bottom:28px;'>Verification System</div>"
        "</div>",
        unsafe_allow_html=True
    )

    page = st.radio("", [
        "Live Verification",
        "Image Verification",
        "Register",
        "Bulk Register",
        "Logs",
        "Users",
    ], label_visibility="collapsed")

    st.markdown(
        "<div style='height:1px;background:#1c1c1c;margin:20px 16px;'></div>",
        unsafe_allow_html=True
    )
    n = count_users()
    st.markdown(
        f"<div style='padding:0 16px;'>"
        f"<div style='font-size:10px;font-weight:600;text-transform:uppercase;"
        f"letter-spacing:0.1em;color:#555;margin-bottom:6px;'>Registered</div>"
        f"<div style='font-size:28px;font-weight:600;color:#ddd;line-height:1;"
        f"margin-bottom:2px;'>{n}</div>"
        f"<div style='font-size:11px;color:#555;'>users in database</div></div>",
        unsafe_allow_html=True
    )


# ── Helpers ───────────────────────────────────────────────────
def page_header(title, desc=""):
    st.markdown(
        f"<div class='pg-title'>{title}</div>"
        f"<div class='pg-sub'>{desc}</div>",
        unsafe_allow_html=True
    )

def stat_card(col, label, value):
    cls = "sc-l-black" if label in ["Verified", "Unknown"] else "sc-l"
    col.markdown(
        f"<div class='sc'><div class='{cls}'>{label}</div>"
        f"<div class='sc-v'>{value}</div></div>",
        unsafe_allow_html=True
    )

def notice(html):
    st.markdown(f"<div class='notice'>{html}</div>", unsafe_allow_html=True)

def divider():
    st.markdown("<div class='div-line'></div>", unsafe_allow_html=True)

def result_row(name, score, kind):
    cls = {"ok":"b-ok", "fail":"b-fail", "dup":"b-dup"}[kind]
    lbl = {"ok":"Verified", "fail":"Unknown", "dup":"Duplicate"}[kind]
    return (f"<tr><td>{name}</td><td>{score:.4f}</td>"
            f"<td><span class='badge {cls}'>{lbl}</span></td></tr>")

def result_table(rows):
    return (
        "<div class='rt-wrap'><table class='rt'>"
        "<thead><tr><th>Identity</th><th>Score</th><th>Status</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></div>"
    )

def should_log(name):
    """
    Returns True only if enough time has passed since the last log
    for this identity. Prevents log spam during continuous webcam feed.
    """
    last = st.session_state.last_log_time.get(name)
    if last is None:
        return True
    return (datetime.now() - last).total_seconds() >= LOG_INTERVAL_SEC


# =============================================================
#  PAGE 1 — Live Verification
# =============================================================
if page == "Live Verification":
    page_header("Live Verification", "Real-time face recognition via webcam")

    users = load_all_users()
    if not users:
        notice("No users registered yet. Go to <b>Register</b> or <b>Bulk Register</b> first.")
        st.stop()

    col_a, col_b, col_c = st.columns([1, 1, 4])
    with col_a:
        if st.button("Start"):
            st.session_state.run_cam        = True
            st.session_state.alert_sent     = False
            st.session_state.recent_entries = {}
            st.session_state.last_log_time  = {}
    with col_b:
        if st.button("Stop"):
            st.session_state.run_cam = False
    with col_c:
        st.markdown(
            f"<div style='font-size:12px;color:#888;padding-top:9px;'>"
            f"{len(users)} user(s) loaded</div>",
            unsafe_allow_html=True
        )

    feed_slot   = st.empty()
    result_slot = st.empty()

    if st.session_state.run_cam:
        cap         = cv2.VideoCapture(0)
        frame_count = 0

        while st.session_state.run_cam:
            ret, frame = cap.read()
            if not ret:
                st.error("Cannot read from webcam.")
                break

            frame_count += 1
            if frame_count % 3 != 0:    # process every 3rd frame only
                continue

            frame = cv2.resize(frame, (640, 480))
            embeddings, boxes = get_all_embeddings(frame)
            rows = ""

            for emb, box in zip(embeddings, boxes):
                name, set_id, score = find_best_match(emb, users)
                ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                now = datetime.now()
                x1, y1, x2, y2 = [int(c) for c in box]

                _, buf      = cv2.imencode(".jpg", frame)
                image_bytes = buf.tobytes()
                face_b64    = base64.b64encode(image_bytes).decode("utf-8")

                if name == "Unknown":
                    color = (50, 50, 200)
                    # Log + alert only every LOG_INTERVAL_SEC seconds
                    if should_log("Unknown"):
                        save_log("Unknown", "Unknown", score, face_b64)
                        save_unknown(face_b64, source="webcam")
                        st.session_state.last_log_time["Unknown"] = now
                        if not st.session_state.alert_sent:
                            alert_unknown_face(timestamp=ts, image_bytes=image_bytes)
                            st.session_state.alert_sent = True
                        cnt = count_unknowns_in_window(config.UNKNOWN_ALERT_WINDOW)
                        if cnt >= config.UNKNOWN_ALERT_COUNT:
                            alert_multiple_unknowns(cnt, config.UNKNOWN_ALERT_WINDOW, image_bytes)
                    rows += result_row("Unknown", score, "fail")

                else:
                    color  = (20, 160, 20)
                    recent = st.session_state.recent_entries
                    dup    = (name in recent and
                              (now - recent[name]).total_seconds() < config.DUPLICATE_WINDOW_SEC)

                    if dup:
                        # Log duplicate only every LOG_INTERVAL_SEC seconds
                        if should_log(f"dup_{name}"):
                            if not st.session_state.alert_sent:
                                alert_duplicate_entry(name, timestamp=ts)
                                st.session_state.alert_sent = True
                            save_log(name, "Duplicate", score, face_b64)
                            st.session_state.last_log_time[f"dup_{name}"] = now
                        rows += result_row(name, score, "dup")
                    else:
                        # First valid entry — log it, then gate further logs
                        if should_log(name):
                            recent[name] = now
                            save_log(name, "Verified", score, face_b64)
                            st.session_state.last_log_time[name] = now
                            # Reset alert flag so next unknown triggers fresh alert
                            st.session_state.alert_sent = False
                        rows += result_row(name, score, "ok")

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.rectangle(frame, (x1, y1 - 26), (x2, y1), color, -1)
                cv2.putText(frame, f"{name}  {score:.2f}", (x1 + 5, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)

            feed_slot.image(frame[:, :, ::-1], channels="RGB",
                            use_container_width=True)
            if rows:
                result_slot.markdown(result_table(rows), unsafe_allow_html=True)

        cap.release()


# =============================================================
#  PAGE 2 — Image Verification
# =============================================================
elif page == "Image Verification":
    page_header("Image Verification", "Upload a photo — single face or group")

    users = load_all_users()
    if not users:
        notice("No users registered yet. Go to <b>Register</b> first.")
        st.stop()

    uploaded = st.file_uploader("", type=["jpg","jpeg","png"],
                                label_visibility="collapsed")

    if uploaded:
        img_pil = Image.open(uploaded).convert("RGB")
        img_bgr = np.array(img_pil)[:, :, ::-1]

        col_l, col_r = st.columns([1, 1], gap="large")
        with col_l:
            st.image(img_pil, use_container_width=True)
        with col_r:
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
            if st.button("Verify"):
                with st.spinner("Detecting and matching..."):
                    results = verify_image(img_bgr, registered_users=users, source="upload")
                if not results:
                    notice("No faces detected in this image.")
                else:
                    rows = "".join(
                        result_row(r["name"], r["score"],
                                   "ok" if r["status"] == "Verified" else "fail")
                        for r in results
                    )
                    st.markdown(result_table(rows), unsafe_allow_html=True)
                    st.markdown(
                        f"<div style='font-size:11px;color:#999;margin-top:10px;'>"
                        f"{len(results)} face(s) detected</div>",
                        unsafe_allow_html=True
                    )


# =============================================================
#  PAGE 3 — Register
# =============================================================
elif page == "Register":
    page_header("Register", "Add a new person to the database")

    col1, col2 = st.columns(2)
    with col1:
        set_id = st.text_input("Person ID", placeholder="e.g. person_42")
    with col2:
        name = st.text_input("Name", placeholder="e.g. Rahul Shah")

    divider()
    method = st.radio("Method", ["Upload images", "Webcam capture"], horizontal=True)

    # ── Upload images ──────────────────────────────────────────
    if method == "Upload images":
        files = st.file_uploader("", type=["jpg","jpeg","png"],
                                 accept_multiple_files=True,
                                 label_visibility="collapsed")
        if files:
            st.markdown(
                f"<div style='font-size:12px;color:#888;margin-top:6px;margin-bottom:4px;'>"
                f"{len(files)} file(s) selected</div>",
                unsafe_allow_html=True
            )

        if st.button("Register"):
            if not set_id or not name:
                st.warning("Fill in Person ID and Name first.")
            elif not files:
                st.warning("Upload at least one image.")
            else:
                embeddings, bar = [], st.progress(0)
                for i, f in enumerate(files):
                    img_bgr = np.array(Image.open(f).convert("RGB"))[:, :, ::-1]
                    emb = get_single_embedding(img_bgr)
                    if emb is not None:
                        embeddings.append(emb)
                    bar.progress((i + 1) / len(files))

                if embeddings:
                    save_user(set_id, name, normalize(np.mean(embeddings, axis=0)))
                    st.success(f"Registered {name} — {len(embeddings)}/{len(files)} images used.")
                else:
                    st.error("No faces detected in any uploaded image.")

    # ── Webcam capture (in-browser via st.camera_input) ────────
    else:
        notice(
            "Take a photo using the camera below. "
            "Capture <b>multiple angles</b> for better accuracy — "
            "each snapshot is added to the registration."
        )

        if not set_id or not name:
            st.warning("Fill in Person ID and Name above before capturing.")
        else:
            # Accumulate captures in session state
            cap_key = f"reg_captures_{set_id}"
            if cap_key not in st.session_state:
                st.session_state[cap_key] = []

            cam_img = st.camera_input("", label_visibility="collapsed")

            if cam_img is not None:
                img_pil = Image.open(cam_img).convert("RGB")
                img_bgr = np.array(img_pil)[:, :, ::-1]
                emb     = get_single_embedding(img_bgr)

                if emb is None:
                    st.warning("No face detected in this capture. Try again.")
                else:
                    st.session_state[cap_key].append(emb)
                    st.success(
                        f"Capture {len(st.session_state[cap_key])} added. "
                        f"Take more or click Register below."
                    )

            count_so_far = len(st.session_state[cap_key])

            col_info, col_btn = st.columns([3, 1])
            with col_info:
                st.markdown(
                    f"<div style='font-size:13px;color:#555;padding-top:10px;'>"
                    f"{count_so_far} capture(s) ready</div>",
                    unsafe_allow_html=True
                )
            with col_btn:
                if st.button("Register", disabled=(count_so_far == 0)):
                    avg = normalize(np.mean(st.session_state[cap_key], axis=0))
                    save_user(set_id, name, avg)
                    st.success(f"Registered {name} with {count_so_far} capture(s).")
                    del st.session_state[cap_key]   # clear captures after saving


# =============================================================
#  PAGE 4 — Bulk Register
# =============================================================
elif page == "Bulk Register":
    page_header("Bulk Register", "Register all persons from a dataset folder at once")

    notice(
        "<b>Expected structure</b><br>"
        "<code>person_1 / ID_1.jpg, Selfie_1.jpg ...</code><br>"
        "<code>person_2 / ...</code><br>"
        "Each subfolder name becomes the person's ID and name."
    )

    col_btn, col_path = st.columns([1, 4])
    with col_btn:
        if st.button("Browse folder"):
            chosen = pick_folder()
            if chosen:
                st.session_state.dataset_path = chosen

    with col_path:
        current = st.session_state.dataset_path
        txt_cls = "" if current else "path-empty"
        display = current if current else "No folder selected"
        st.markdown(
            f"<div style='padding-top:6px;'>"
            f"<div class='path-box {txt_cls}'>{display}</div></div>",
            unsafe_allow_html=True
        )

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    skip = st.checkbox("Skip already registered persons", value=True)

    if st.button("Start Registration"):
        path = st.session_state.dataset_path
        if not path:
            st.warning("Select a folder first using Browse folder.")
        else:
            with st.spinner("Processing dataset..."):
                s = register_bulk(dataset_path=path, skip_existing=skip)

            c1, c2, c3, c4 = st.columns(4)
            stat_card(c1, "Total",      s["total"])
            stat_card(c2, "Registered", s["registered"])
            stat_card(c3, "Skipped",    s["skipped"])
            stat_card(c4, "Failed",     s["failed"])


# =============================================================
#  PAGE 5 — Logs
# =============================================================
elif page == "Logs":
    page_header("Logs", "Verification event history")

    logs = get_recent_logs(50)

    if not logs:
        notice("No events recorded yet.")
    else:
        df = pd.DataFrame(logs)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")

        c1, c2, c3 = st.columns(3)
        stat_card(c1, "Showing",  len(df))
        stat_card(c2, "Verified", len(df[df["status"] == "Verified"]))
        stat_card(c3, "Unknown",  len(df[df["status"] == "Unknown"]))

        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

        st.dataframe(
            df[["timestamp","identity","status","score"]].rename(columns={
                "timestamp":"Timestamp","identity":"Identity",
                "status":"Status","score":"Score"
            }),
            use_container_width=True,
            hide_index=True
        )

        # snapshot preview
        log_options = [
            f"{log.get('timestamp','')}  —  {log.get('identity','')}  ({log.get('status','')})"
            for log in logs
        ]
        selected_idx = st.selectbox("Preview snapshot", range(len(log_options)),
                                     format_func=lambda i: log_options[i])
        entry = logs[selected_idx]
        if entry.get("image_b64"):
            img_bytes = base64.b64decode(entry["image_b64"])
            st.image(img_bytes,
                     caption=f"{entry.get('identity')}  —  {entry.get('timestamp')}",
                     use_container_width=False, width=360)
        else:
            st.markdown(
                "<div style='font-size:12px;color:#999;margin-top:6px;'>"
                "No snapshot for this entry.</div>",
                unsafe_allow_html=True
            )

        divider()
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, "logs.csv", "text/csv")


# =============================================================
#  PAGE 6 — Users
# =============================================================
elif page == "Users":
    page_header("Users", "Manage registered users")

    users = load_all_users()
    if not users:
        notice("No users registered yet.")
    else:
        df = pd.DataFrame([{"ID": u["set_id"], "Name": u["name"]} for u in users])
        st.dataframe(df, use_container_width=True, hide_index=True)

        divider()

        col1, col2 = st.columns([3, 1])
        with col1:
            del_id = st.text_input("", placeholder="Enter Person ID to delete",
                                   label_visibility="collapsed")
        with col2:
            st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)
            if st.button("Delete"):
                if not del_id:
                    st.warning("Enter an ID.")
                elif delete_user(del_id):
                    st.success(f"Deleted: {del_id}")
                    st.rerun()
                else:
                    st.error(f"ID not found: {del_id}")