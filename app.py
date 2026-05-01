import os
import time
import tempfile
import json
import cv2
import pandas as pd
import streamlit as st

from analyzer import VideoAnalyzer
from graph_utils import plot_live_graph
from utils import ensure_dir

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
TEMP_DIR = os.path.join(OUTPUT_DIR, "temp")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")

st.set_page_config(page_title="CV Module - Real-Time Video Analysis", layout="wide")

ensure_dir(OUTPUT_DIR)
ensure_dir(TEMP_DIR)
ensure_dir(REPORTS_DIR)


def init_state():
    defaults = {
        "video_stop_requested": False,
        "camera_stop_requested": False,
        "video_report": None,
        "camera_report": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


def save_uploaded_file(uploaded_file):
    suffix = "." + uploaded_file.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=TEMP_DIR) as tmp_file:
        tmp_file.write(uploaded_file.read())
        return tmp_file.name


def format_metrics(record):
    emotions = record.get("emotions", {})
    sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)

    html = f"""
    <div style="padding:15px;border-radius:10px;border:1px solid #444;">
        <h4>Current Frame Metrics</h4>
        <p><b>Frame:</b> {record.get('frame_index')}</p>
        <p><b>Timestamp:</b> {record.get('timestamp_sec')} sec</p>
        <p><b>Dominant Emotion:</b> {record.get('dominant_emotion')}</p>
        <p><b>Eye Contact Score:</b> {record.get('eye_contact_score')}</p>
        <p><b>Head Pose Score:</b> {record.get('head_pose_score')}</p>
        <p><b>Posture Score:</b> {record.get('posture_score')}</p>
        <p><b>Overall Visual Score:</b> {record.get('overall_visual_score')}</p>
        <hr>
        <h5>Emotion Scores</h5>
    """
    for emotion, score in sorted_emotions:
        html += f"<p>{emotion}: {round(score, 2)}</p>"
    html += "</div>"
    return html


def build_feature_stats_df(report):
    rows = []
    for _, item in report["statistics"]["feature_statistics"].items():
        rows.append({
            "Metric": item["label"],
            "Average": item["average"],
            "Max": item["max"],
            "Min": item["min"],
        })
    return pd.DataFrame(rows)


def build_emotion_stats_df(report):
    rows = []
    for emotion, item in report["statistics"]["emotion_statistics"].items():
        rows.append({
            "Emotion": emotion,
            "Average": item["average"],
            "Max": item["max"],
            "Min": item["min"],
        })
    return pd.DataFrame(rows)


def render_report_block(report, title_prefix):
    if report is None:
        return

    summary = report["summary"]
    feature_df = build_feature_stats_df(report)
    emotion_df = build_emotion_stats_df(report)

    st.markdown("---")
    st.header(f"{title_prefix} Final Summary")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Frames", summary["total_frames_processed"])
    c2.metric("Dominant Emotion", summary["dominant_emotion_overall"])
    c3.metric("Avg Eye Contact", summary["average_eye_contact_score"])
    c4.metric("Avg Overall Score", summary["average_overall_visual_score"])

    st.subheader("Small Summary")
    st.write(
        f"Source: {summary['source_name']} | "
        f"Type: {summary['source_type']} | "
        f"Stopped Early: {summary['stopped_early']}"
    )

    st.subheader("Feature Statistics Table")
    st.dataframe(feature_df, use_container_width=True)

    st.subheader("Emotion Statistics Table")
    st.dataframe(emotion_df, use_container_width=True)

    json_data = json.dumps(report, indent=4)
    feature_csv_data = feature_df.to_csv(index=False)
    emotion_csv_data = emotion_df.to_csv(index=False)

    st.subheader("Download Reports")
    d1, d2, d3 = st.columns(3)

    with d1:
        st.download_button(
            label=f"Download {title_prefix} JSON Report",
            data=json_data,
            file_name=f"{title_prefix.lower()}_report.json",
            mime="application/json",
            use_container_width=True,
        )

    with d2:
        st.download_button(
            label=f"Download {title_prefix} Feature CSV",
            data=feature_csv_data,
            file_name=f"{title_prefix.lower()}_feature_stats.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with d3:
        st.download_button(
            label=f"Download {title_prefix} Emotion CSV",
            data=emotion_csv_data,
            file_name=f"{title_prefix.lower()}_emotion_stats.csv",
            mime="text/csv",
            use_container_width=True,
        )


st.title("🧩 Computer Vision Module")
st.subheader("Video Upload Analysis + Live Camera Analysis + JSON Report + CSV Stats")

tab1, tab2 = st.tabs(["📁 Upload Video", "📷 Camera Live"])


with tab1:
    st.markdown("### Upload Video Analysis")

    uploaded_video = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov", "mkv"],
        key="uploaded_video_tab",
    )

    top1, top2, top3 = st.columns([1, 1, 1])

    with top1:
        max_frames = st.number_input(
            "Max frames to process (0 = full video)",
            min_value=0,
            max_value=100000,
            value=0,
            step=50,
            key="video_max_frames",
        )

    with top2:
        start_video = st.button("Start Analysis", use_container_width=True, key="start_video_btn")

    with top3:
        stop_video = st.button("Stop Analysis", use_container_width=True, key="stop_video_btn")

    if stop_video:
        st.session_state.video_stop_requested = True

    left_col, right_col = st.columns([1.5, 1])

    video_placeholder = left_col.empty()
    graph_placeholder = right_col.empty()
    metrics_placeholder = right_col.empty()
    progress_placeholder = st.empty()
    status_placeholder = st.empty()

    if uploaded_video is not None:
        st.video(uploaded_video)

    if uploaded_video is not None and start_video:
        try:
            st.session_state.video_stop_requested = False
            st.session_state.video_report = None

            temp_video_path = save_uploaded_file(uploaded_video)
            analyzer = VideoAnalyzer(output_dir=REPORTS_DIR)

            progress_bar = progress_placeholder.progress(0.0)
            status_placeholder.info("Video analysis started...")

            def frame_callback(frame, record, all_records):
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                graph_img = plot_live_graph(all_records)
                graph_placeholder.image(graph_img, caption="Live Graph", use_container_width=True)

                metrics_placeholder.markdown(format_metrics(record), unsafe_allow_html=True)
                time.sleep(0.01)

            def progress_callback(value):
                progress_bar.progress(float(min(max(value, 0.0), 1.0)))

            def should_stop():
                return st.session_state.video_stop_requested

            max_frames_value = None if max_frames == 0 else int(max_frames)

            _, report = analyzer.process_video(
                temp_video_path,
                frame_callback=frame_callback,
                progress_callback=progress_callback,
                should_stop_callback=should_stop,
                max_frames=max_frames_value,
            )

            st.session_state.video_report = report

            if report["summary"]["stopped_early"]:
                status_placeholder.warning("Video analysis stopped early. Partial report generated.")
            else:
                status_placeholder.success("Video analysis completed successfully.")

            progress_bar.progress(1.0)

        except Exception as e:
            status_placeholder.error(f"Error during video analysis: {e}")
            st.exception(e)

    render_report_block(st.session_state.video_report, "Video")


with tab2:
    st.markdown("### Live Camera Analysis")

    left_cam, right_cam = st.columns([1.5, 1])

    cam_video_placeholder = left_cam.empty()

    with right_cam:
        camera_index = st.number_input(
            "Camera Index",
            min_value=0,
            max_value=5,
            value=0,
            step=1,
            key="camera_index"
        )
        camera_max_frames = st.number_input(
            "Frames to capture",
            min_value=30,
            max_value=5000,
            value=300,
            step=30,
            key="camera_max_frames"
        )
        target_fps = st.number_input(
            "Target FPS",
            min_value=1,
            max_value=30,
            value=15,
            step=1,
            key="camera_target_fps"
        )

        start_camera = st.button("Start Camera Analysis", use_container_width=True, key="start_camera_btn")
        stop_camera = st.button("Stop Camera Analysis", use_container_width=True, key="stop_camera_btn")

        cam_graph_placeholder = st.empty()
        cam_metrics_placeholder = st.empty()

    cam_status_placeholder = st.empty()

    if stop_camera:
        st.session_state.camera_stop_requested = True

    if start_camera:
        try:
            st.session_state.camera_stop_requested = False
            st.session_state.camera_report = None

            analyzer = VideoAnalyzer(output_dir=REPORTS_DIR)
            cam_status_placeholder.info("Camera analysis started...")

            def cam_frame_callback(frame, record, all_records):
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cam_video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                graph_img = plot_live_graph(all_records)
                cam_graph_placeholder.image(graph_img, caption="Camera Live Graph", use_container_width=True)

                cam_metrics_placeholder.markdown(format_metrics(record), unsafe_allow_html=True)

            def cam_should_stop():
                return st.session_state.camera_stop_requested

            _, report = analyzer.process_webcam(
                camera_index=int(camera_index),
                frame_callback=cam_frame_callback,
                should_stop_callback=cam_should_stop,
                max_frames=int(camera_max_frames),
                target_fps=int(target_fps),
            )

            st.session_state.camera_report = report

            if report["summary"]["stopped_early"]:
                cam_status_placeholder.warning("Camera analysis stopped early. Partial report generated.")
            else:
                cam_status_placeholder.success("Camera analysis completed successfully.")

        except Exception as e:
            cam_status_placeholder.error(f"Error during camera analysis: {e}")
            st.exception(e)

    render_report_block(st.session_state.camera_report, "Camera")