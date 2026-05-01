import os
import cv2
import json
import time
import numpy as np

from utils import EMOTIONS, ensure_dir, now_string, normalize_emotion_scores

DeepFace = None

#mediapipe for face mesh and pose estimation, opencv for video processing, deepface for emotion analysis
class VideoAnalyzer:
    def __init__(self, output_dir=None):
        global DeepFace

        base_dir = os.path.dirname(os.path.abspath(__file__))

        if output_dir is None:
            output_dir = os.path.join(base_dir, "output", "reports")

        self.base_dir = base_dir
        self.output_dir = os.path.abspath(output_dir)
        ensure_dir(self.output_dir)

        try:
            from mediapipe.python.solutions import face_mesh, pose, drawing_utils, drawing_styles
        except Exception:
            import mediapipe as mp
            face_mesh = mp.solutions.face_mesh
            pose = mp.solutions.pose
            drawing_utils = mp.solutions.drawing_utils
            drawing_styles = mp.solutions.drawing_styles

        self.mp_face_mesh = face_mesh
        self.mp_pose = pose
        self.mp_drawing = drawing_utils
        self.mp_drawing_styles = drawing_styles

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.records = []
        self.last_emotion_scores = {emotion: 0.0 for emotion in EMOTIONS}

        if DeepFace is None:
            from deepface import DeepFace as DF
            DeepFace = DF

        print(f"[VideoAnalyzer] base_dir   = {self.base_dir}")
        print(f"[VideoAnalyzer] output_dir = {self.output_dir}")
#capture frame data and optimization using previous result
    def reset_records(self):
        self.records = []
        self.last_emotion_scores = {emotion: 0.0 for emotion in EMOTIONS}
#get face bounding box from landmarks
    def get_face_bbox_from_landmarks(self, frame, face_landmarks):
        h, w, _ = frame.shape
        xs = [int(lm.x * w) for lm in face_landmarks.landmark]
        ys = [int(lm.y * h) for lm in face_landmarks.landmark]

        x1 = max(min(xs) - 20, 0)
        y1 = max(min(ys) - 20, 0)
        x2 = min(max(xs) + 20, w)
        y2 = min(max(ys) + 20, h)
        return x1, y1, x2, y2
#uses deepface for emotion analysis
    def analyze_emotion(self, face_crop):
        global DeepFace
        try:
            result = DeepFace.analyze(
                img_path=face_crop,
                actions=["emotion"],
                enforce_detection=False,
                detector_backend="opencv",
                silent=True,
            )

            if isinstance(result, list):
                result = result[0]

            emotions = result.get("emotion", {})
            return normalize_emotion_scores(emotions)
        except Exception as e:
            print(f"[Emotion] DeepFace failed: {e}")
            return {emotion: 0.0 for emotion in EMOTIONS}
#Calculates:Pitch (up/down) Yaw (left/right) Roll (tilt) for headpose
    def estimate_head_pose(self, frame, face_landmarks):
        h, w, _ = frame.shape

        image_points = []
        model_points = []

        selected = {
            1: (0.0, 0.0, 0.0),
            33: (-30.0, -30.0, -30.0),
            263: (30.0, -30.0, -30.0),
            61: (-25.0, 30.0, -20.0),
            291: (25.0, 30.0, -20.0),
            199: (0.0, 60.0, -50.0),
        }

        try:
            for idx, model_pt in selected.items():
                lm = face_landmarks.landmark[idx]
                x, y = int(lm.x * w), int(lm.y * h)
                image_points.append((x, y))
                model_points.append(model_pt)

            image_points = np.array(image_points, dtype="double")
            model_points = np.array(model_points, dtype="double")

            focal_length = w
            center = (w / 2, h / 2)
            camera_matrix = np.array(
                [[focal_length, 0, center[0]],
                 [0, focal_length, center[1]],
                 [0, 0, 1]],
                dtype="double",
            )

            dist_coeffs = np.zeros((4, 1))

            success, rotation_vector, translation_vector = cv2.solvePnP(  #Converts image points → 3D orientation
                model_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )

            if not success:
                return {"pitch": 0.0, "yaw": 0.0, "roll": 0.0, "score": 0.0}

            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            pose_matrix = cv2.hconcat((rotation_matrix, translation_vector))
            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_matrix)

            pitch = float(euler_angles[0][0])
            yaw = float(euler_angles[1][0])
            roll = float(euler_angles[2][0])

            deviation = abs(pitch) + abs(yaw) + abs(roll) * 0.5
            score = max(0.0, 100.0 - min(deviation, 100.0))

            return {
                "pitch": round(pitch, 2),
                "yaw": round(yaw, 2),
                "roll": round(roll, 2),
                "score": round(score, 2),
            }
        except Exception as e:
            print(f"[HeadPose] failed: {e}")
            return {"pitch": 0.0, "yaw": 0.0, "roll": 0.0, "score": 0.0}

    def estimate_eye_contact(self, frame, face_landmarks):
        h, w, _ = frame.shape

        left_iris = [468, 469, 470, 471, 472]
        right_iris = [473, 474, 475, 476, 477]

        left_eye_outer = 33
        left_eye_inner = 133
        right_eye_inner = 362
        right_eye_outer = 263
# traces the eye skeleton
        try:
            def avg_point(indices):
                pts = []
                for idx in indices:
                    lm = face_landmarks.landmark[idx]
                    pts.append((lm.x * w, lm.y * h))
                x = sum(p[0] for p in pts) / len(pts)
                y = sum(p[1] for p in pts) / len(pts)
                return x, y

            left_iris_center = avg_point(left_iris)
            right_iris_center = avg_point(right_iris)

            l_outer = face_landmarks.landmark[left_eye_outer]
            l_inner = face_landmarks.landmark[left_eye_inner]
            r_inner = face_landmarks.landmark[right_eye_inner]
            r_outer = face_landmarks.landmark[right_eye_outer]

            l_outer_pt = (l_outer.x * w, l_outer.y * h)
            l_inner_pt = (l_inner.x * w, l_inner.y * h)
            r_inner_pt = (r_inner.x * w, r_inner.y * h)
            r_outer_pt = (r_outer.x * w, r_outer.y * h)
#tracks movement of eye skeleton
            def iris_ratio(iris_center, outer_pt, inner_pt):
                eye_width = abs(inner_pt[0] - outer_pt[0]) + 1e-6
                pos = abs(iris_center[0] - outer_pt[0]) / eye_width
                return pos

            left_ratio = iris_ratio(left_iris_center, l_outer_pt, l_inner_pt)
            right_ratio = iris_ratio(right_iris_center, r_outer_pt, r_inner_pt)

            left_score = 100 - min(abs(left_ratio - 0.5) * 200, 100)
            right_score = 100 - min(abs(right_ratio - 0.5) * 200, 100)
            eye_contact_score = (left_score + right_score) / 2

            return {
                "left_eye_ratio": round(left_ratio, 3),
                "right_eye_ratio": round(right_ratio, 3),
                "score": round(eye_contact_score, 2),
            }
        except Exception as e:
            print(f"[EyeContact] failed: {e}")
            return {"left_eye_ratio": 0.0, "right_eye_ratio": 0.0, "score": 0.0}

    def estimate_posture(self, frame, pose_landmarks):
        h, w, _ = frame.shape
        try:
            lm = pose_landmarks.landmark

            left_shoulder = (
                lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w,
                lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h,
            )
            right_shoulder = (
                lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
                lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h,
            )
            nose = (
                lm[self.mp_pose.PoseLandmark.NOSE.value].x * w,
                lm[self.mp_pose.PoseLandmark.NOSE.value].y * h,
            )
            mid_shoulder = (
                (left_shoulder[0] + right_shoulder[0]) / 2,
                (left_shoulder[1] + right_shoulder[1]) / 2,
            )

            shoulder_tilt = abs(left_shoulder[1] - right_shoulder[1])
            alignment_offset = abs(nose[0] - mid_shoulder[0])

            tilt_penalty = min(shoulder_tilt / 2, 50)
            align_penalty = min(alignment_offset / 4, 50)
            posture_score = max(0.0, 100.0 - (tilt_penalty + align_penalty))

            return {
                "shoulder_tilt": round(float(shoulder_tilt), 2),
                "alignment_offset": round(float(alignment_offset), 2),
                "score": round(float(posture_score), 2),
            }
        except Exception as e:
            print(f"[Posture] failed: {e}")
            return {"shoulder_tilt": 0.0, "alignment_offset": 0.0, "score": 0.0}
#draws the face mesh
    def draw_face_landmarks(self, frame, face_landmarks):
        try:
            self.mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style(),
            )
        except Exception:
            pass
#draws the pose skeleton
    def draw_pose_landmarks(self, frame, pose_landmarks):
        try:
            self.mp_drawing.draw_landmarks(
                frame,
                pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
        except Exception:
            pass
#shows metrics on the top left of screen, with color coding based on score thresholds
    def overlay_metrics(self, frame, record):
        x, y = 20, 30
        line_gap = 28

        cv2.putText(frame, f"Emotion: {record['dominant_emotion']}", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        y += line_gap
        cv2.putText(frame, f"Eye Contact: {record['eye_contact_score']:.2f}", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
        y += line_gap
        cv2.putText(frame, f"Head Pose: {record['head_pose_score']:.2f}", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 255), 2)
        y += line_gap
        cv2.putText(frame, f"Posture: {record['posture_score']:.2f}", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
        y += line_gap
        cv2.putText(frame, f"Overall: {record['overall_visual_score']:.2f}", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#calculate all the metrics and scores for each frame, and summary statistics at the end
    def process_single_frame(self, frame, frame_index=0, fps=25.0):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_results = self.face_mesh.process(rgb)
        pose_results = self.pose.process(rgb)

        emotion_scores = {emotion: 0.0 for emotion in EMOTIONS}
        eye_contact_score = 0.0
        head_pose_score = 0.0
        posture_score = 0.0
        head_pose_data = {"pitch": 0.0, "yaw": 0.0, "roll": 0.0, "score": 0.0}
        eye_contact_data = {"left_eye_ratio": 0.0, "right_eye_ratio": 0.0, "score": 0.0}
        posture_data = {"shoulder_tilt": 0.0, "alignment_offset": 0.0, "score": 0.0}
        face_detected = False
        pose_detected = False

        if face_results.multi_face_landmarks:
            face_detected = True
            face_landmarks = face_results.multi_face_landmarks[0]

            self.draw_face_landmarks(frame, face_landmarks)

            x1, y1, x2, y2 = self.get_face_bbox_from_landmarks(frame, face_landmarks)
            face_crop = frame[y1:y2, x1:x2]

            if face_crop is not None and face_crop.size > 0:
                if frame_index % 5 == 0:
                    emotion_scores = self.analyze_emotion(face_crop)
                    self.last_emotion_scores = emotion_scores
                else:
                    emotion_scores = self.last_emotion_scores

            eye_contact_data = self.estimate_eye_contact(frame, face_landmarks)
            eye_contact_score = eye_contact_data["score"]

            head_pose_data = self.estimate_head_pose(frame, face_landmarks)
            head_pose_score = head_pose_data["score"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        else:
            emotion_scores = self.last_emotion_scores

        if pose_results.pose_landmarks:
            pose_detected = True
            self.draw_pose_landmarks(frame, pose_results.pose_landmarks)
            posture_data = self.estimate_posture(frame, pose_results.pose_landmarks)
            posture_score = posture_data["score"]

        overall_visual_score = round((eye_contact_score + head_pose_score + posture_score) / 3.0, 2)
        dominant_emotion = max(emotion_scores, key=emotion_scores.get)
        timestamp_sec = round(frame_index / fps, 2)

        record = {
            "frame_index": int(frame_index),
            "timestamp_sec": float(timestamp_sec),
            "face_detected": bool(face_detected),
            "pose_detected": bool(pose_detected),
            "dominant_emotion": str(dominant_emotion),
            "emotions": {k: float(v) for k, v in emotion_scores.items()},
            "eye_contact": eye_contact_data,
            "head_pose": head_pose_data,
            "posture": posture_data,
            "eye_contact_score": round(float(eye_contact_score), 2),
            "head_pose_score": round(float(head_pose_score), 2),
            "posture_score": round(float(posture_score), 2),
            "overall_visual_score": round(float(overall_visual_score), 2),
        }

        self.records.append(record)
        self.overlay_metrics(frame, record)
        return frame, record

    def compute_statistics(self):
        if not self.records:
            zero_feature_stats = {
                "eye_contact_score": {"label": "Eye Contact Score", "average": 0.0, "max": 0.0, "min": 0.0},
                "head_pose_score": {"label": "Head Pose Score", "average": 0.0, "max": 0.0, "min": 0.0},
                "posture_score": {"label": "Posture Score", "average": 0.0, "max": 0.0, "min": 0.0},
                "overall_visual_score": {"label": "Overall Visual Score", "average": 0.0, "max": 0.0, "min": 0.0},
            }
            zero_emotion_stats = {
                emotion: {"average": 0.0, "max": 0.0, "min": 0.0}
                for emotion in EMOTIONS
            }
            return zero_feature_stats, zero_emotion_stats

        feature_keys = {
            "eye_contact_score": "Eye Contact Score",
            "head_pose_score": "Head Pose Score",
            "posture_score": "Posture Score",
            "overall_visual_score": "Overall Visual Score",
        }

        feature_stats = {}
        for key, label in feature_keys.items():
            values = [float(r.get(key, 0.0)) for r in self.records]
            feature_stats[key] = {
                "label": label,
                "average": round(sum(values) / len(values), 2),
                "max": round(max(values), 2),
                "min": round(min(values), 2),
            }

        emotion_stats = {}
        for emotion in EMOTIONS:
            values = [float(r.get("emotions", {}).get(emotion, 0.0)) for r in self.records]
            emotion_stats[emotion] = {
                "average": round(sum(values) / len(values), 2),
                "max": round(max(values), 2),
                "min": round(min(values), 2),
            }

        return feature_stats, emotion_stats

    def _write_json_report(self, payload, report_id):
        ensure_dir(self.output_dir)
        json_path = os.path.join(self.output_dir, f"cv_report_{report_id}.json")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4)

        print(f"[Report] saved to: {json_path}")
        print(f"[Report] exists? {os.path.exists(json_path)}")
        return json_path

    def build_final_report(self, source_name, fps, source_type, stopped_early=False):
        report_id = now_string()

        if not self.records:
            feature_stats, emotion_stats = self.compute_statistics()

            payload = {
                "module": "computer_vision",
                "version": "1.0",
                "generated_at": report_id,
                "summary": {
                    "source_name": source_name,
                    "source_type": source_type,
                    "fps": float(fps),
                    "total_frames_processed": 0,
                    "stopped_early": bool(stopped_early),
                    "dominant_emotion_overall": "neutral",
                    "average_emotions": {emotion: 0.0 for emotion in EMOTIONS},
                    "average_eye_contact_score": 0.0,
                    "average_head_pose_score": 0.0,
                    "average_posture_score": 0.0,
                    "average_overall_visual_score": 0.0,
                },
                "statistics": {
                    "feature_statistics": feature_stats,
                    "emotion_statistics": emotion_stats,
                },
                "timeline": [],
            }

            json_path = self._write_json_report(payload, report_id)
            return json_path, payload

        avg_emotions = {}
        for emotion in EMOTIONS:
            avg_emotions[emotion] = round(
                sum(float(r["emotions"][emotion]) for r in self.records) / len(self.records), 2
            )

        feature_stats, emotion_stats = self.compute_statistics()

        summary = {
            "source_name": source_name,
            "source_type": source_type,
            "fps": float(fps),
            "total_frames_processed": len(self.records),
            "stopped_early": bool(stopped_early),
            "dominant_emotion_overall": max(avg_emotions, key=avg_emotions.get),
            "average_emotions": avg_emotions,
            "average_eye_contact_score": feature_stats["eye_contact_score"]["average"],
            "average_head_pose_score": feature_stats["head_pose_score"]["average"],
            "average_posture_score": feature_stats["posture_score"]["average"],
            "average_overall_visual_score": feature_stats["overall_visual_score"]["average"],
        }

        payload = {
            "module": "computer_vision",
            "version": "1.0",
            "generated_at": report_id,
            "summary": summary,
            "statistics": {
                "feature_statistics": feature_stats,
                "emotion_statistics": emotion_stats,
            },
            "timeline": self.records,
        }

        json_path = self._write_json_report(payload, report_id)
        return json_path, payload
#uploaded video code 
    def process_video(self, video_path, frame_callback=None, progress_callback=None, should_stop_callback=None, max_frames=None):
        self.reset_records()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and fps > 0 else 25.0

        frame_index = 0
        stopped_early = False

        try:
            while True:
                if should_stop_callback and should_stop_callback():
                    stopped_early = True
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                if max_frames is not None and frame_index >= max_frames:
                    stopped_early = True
                    break

                processed_frame, record = self.process_single_frame(frame, frame_index=frame_index, fps=fps)

                if frame_callback:
                    frame_callback(processed_frame, record, self.records)

                if progress_callback and total_frames > 0:
                    progress_callback((frame_index + 1) / total_frames)

                frame_index += 1
        finally:
            cap.release()

        return self.build_final_report(
            source_name=os.path.basename(video_path),
            fps=fps,
            source_type="uploaded_video",
            stopped_early=stopped_early,
        )
# webcame code 
    def process_webcam(self, camera_index=0, frame_callback=None, should_stop_callback=None, max_frames=300, target_fps=15):
        self.reset_records()

        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise ValueError("Could not access webcam.")

        fps = float(target_fps if target_fps > 0 else 15)
        frame_index = 0
        frame_delay = 1.0 / fps
        stopped_early = False

        try:
            while frame_index < max_frames:
                if should_stop_callback and should_stop_callback():
                    stopped_early = True
                    break

                start_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    break

                processed_frame, record = self.process_single_frame(frame, frame_index=frame_index, fps=fps)

                if frame_callback:
                    frame_callback(processed_frame, record, self.records)

                frame_index += 1

                elapsed = time.time() - start_time
                sleep_time = max(0.0, frame_delay - elapsed)
                time.sleep(sleep_time)
        finally:
            cap.release()

        return self.build_final_report(
            source_name=f"webcam_{camera_index}",
            fps=fps,
            source_type="webcam",
            stopped_early=stopped_early,
        )