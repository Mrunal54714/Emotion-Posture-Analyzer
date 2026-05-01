# Graph utilities for the CV module
import io
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

EMOTION_COLORS = {
    "angry": "red",
    "disgust": "green",
    "fear": "purple",
    "happy": "orange",
    "sad": "blue",
    "surprise": "pink",
    "neutral": "gray",
}

FEATURE_COLORS = {
    "eye_contact_score": "cyan",
    "head_pose_score": "brown",
    "posture_score": "black",
    "overall_visual_score": "gold",
}


def plot_live_graph(data_records):
    """
    Returns graph as PIL image.
    """
    if not data_records:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.set_title("Live Analysis Graph")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 100)
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)

    x = list(range(len(data_records)))

    fig, ax = plt.subplots(figsize=(14, 6))

    # Emotion lines
    for emotion, color in EMOTION_COLORS.items():
        y = [record["emotions"].get(emotion, 0) for record in data_records]
        ax.plot(x, y, label=emotion, color=color, linewidth=1.8)

    # Feature lines
    for feature, color in FEATURE_COLORS.items():
        y = [record.get(feature, 0) for record in data_records]
        ax.plot(x, y, label=feature, color=color, linestyle="--", linewidth=2)

    ax.set_title("Real-Time Emotion & Feature Scores")
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)