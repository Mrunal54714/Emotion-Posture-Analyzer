# Utility functions for the CV module
import cv2
import math
import numpy as np
from datetime import datetime


EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

#creating file to store session data   
def ensure_dir(path):
    import os
    os.makedirs(path, exist_ok=True)

#unique filename based on timestamp  cv_module/output/report
def now_string():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# show all emotions with 0.0 if missing, and ensure values are floats
def normalize_emotion_scores(emotion_dict):
    """
    Ensure all required emotions exist and values are floats.
    """
    normalized = {emotion: 0.0 for emotion in EMOTIONS}
    if emotion_dict:
        for key, value in emotion_dict.items():
            k = key.lower()
            if k in normalized:
                normalized[k] = float(value)
    return normalized

# calculate eye distance, mouth opening
def calculate_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

# calculate angle between three points (e.g., shoulder-elbow-wrist for arm angle)
def calculate_angle(a, b, c):
    """
    Returns angle ABC in degrees
    a, b, c are (x, y)
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  #keeping values in fixed range
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

# color coding based on score
def get_text_color(score):
    """
    Utility for optional color coding
    """
    if score >= 75:
        return (0, 255, 0) #green
    elif score >= 40:
        return (0, 255, 255) #yellow
    return (0, 0, 255) #red

# resize fram based on webcam feed, maintain aspect ratio
def resize_frame(frame, width=900):
    h, w = frame.shape[:2]
    ratio = width / float(w)
    height = int(h * ratio)
    return cv2.resize(frame, (width, height))

# safely convert to float, return default if fails  
def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default