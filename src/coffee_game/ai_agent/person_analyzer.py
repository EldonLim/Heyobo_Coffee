import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from deepface import DeepFace
import urllib.request
import os
import numpy as np
import json
from collections import Counter, deque


class PersonAnalyzer:
    """
    A class to analyze person attributes from video/camera feed.
    Detects: emotion, gender, age, glasses, and shirt color.
    """
    
    MODEL_PATH = "blaze_face_short_range.tflite"
    POSE_MODEL_PATH = "pose_landmarker_lite.task"
    
    def __init__(self, analyze_interval=5, output_interval=30):
        """
        Initialize the PersonAnalyzer.
        
        Args:
            analyze_interval: Number of frames between each analysis (default: 5)
            output_interval: Number of frames to aggregate before output (default: 30 = 1 second)
        """
        self.analyze_interval = analyze_interval
        self.output_interval = output_interval
        self.frame_count = 0
        self.current_face_data = {}
        self.current_shirt_color = None
        
        # History buffer to store traits over output_interval frames
        # Each key is a person index, value is a dict of trait lists
        self.trait_history = {}
        self.aggregated_output = None
        
        # Download models if needed
        self._download_models()
        
        # Initialize detectors
        self._init_detectors()
        
        # Initialize DeepLabV3+ for shirt segmentation
        self._init_deeplab()

    def _download_models(self):
        """Download required models if not present."""
        if not os.path.exists(self.MODEL_PATH):
            print("Downloading face detection model...")
            url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
            urllib.request.urlretrieve(url, self.MODEL_PATH)
            print("Model downloaded!")
        
        if not os.path.exists(self.POSE_MODEL_PATH):
            print("Downloading pose detection model...")
            url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
            urllib.request.urlretrieve(url, self.POSE_MODEL_PATH)
            print("Pose model downloaded!")
    
    def _init_detectors(self):
        """Initialize MediaPipe detectors."""
        # Face detector
        base_options = python.BaseOptions(model_asset_path=self.MODEL_PATH)
        options = vision.FaceDetectorOptions(base_options=base_options)
        self.face_detector = vision.FaceDetector.create_from_options(options)
        
        # Pose detector
        pose_base_options = python.BaseOptions(model_asset_path=self.POSE_MODEL_PATH)
        pose_options = vision.PoseLandmarkerOptions(base_options=pose_base_options)
        self.pose_detector = vision.PoseLandmarker.create_from_options(pose_options)
    
    def _init_deeplab(self):
        """Initialize DeepLabV3+ model for shirt segmentation."""
        try:
            from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
            import torch
            self.deeplab_processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640")
            self.deeplab_model = AutoModelForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640")
            self.deeplab_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.deeplab_model.to(self.deeplab_device)
        except Exception as e:
            import traceback
            print("DeepLabV3+ initialization failed:")
            traceback.print_exc()
            self.deeplab_model = None

    @staticmethod
    def _get_color_name(hue, saturation, value):
        """Map HSV values to color name."""
        if saturation < 30:
            if value < 50:
                return "black"
            elif value < 180:
                return "gray"
            else:
                return "white"
        
        if value < 50:
            return "black"
        
        if hue < 10 or hue >= 160:
            return "red"
        elif hue < 25:
            return "orange"
        elif hue < 35:
            return "yellow"
        elif hue < 85:
            return "green"
        elif hue < 130:
            return "blue"
        elif hue < 160:
            return "purple"
        
        return "unknown"
    
    def _fallback_chest_roi(self, h, w):
        x1 = int(w * 0.35)
        x2 = int(w * 0.65)
        y1 = int(h * 0.40)
        y2 = int(h * 0.70)
        return x1, x2, y1, y2
    
    def _detect_shirt_color(self, frame, landmarks, h, w):
        # Use DeepLabV3+ to segment shirt/upper clothes
        if not hasattr(self, "deeplab_model") or self.deeplab_model is None:
            print("DeepLabV3+ not available, falling back to HSV method.")
            return "unknown"

        import torch
        from PIL import Image
        import cv2
        import numpy as np

        # Convert BGR to RGB and to PIL Image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        inputs = self.deeplab_processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self.deeplab_device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.deeplab_model(**inputs)
        logits = outputs.logits
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=pil_img.size[::-1],
            mode="bilinear",
            align_corners=False
        )
        seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
        print(f"[DEBUG] Segmentation unique values: {np.unique(seg)}")

        # ADE20K class indices for upper clothes/shirt/jacket
        # 5: 'upper clothes', 6: 'dress', 7: 'coat', 13: 'shirt', 15: 'jacket'
        shirt_classes = [5, 6, 7, 13, 15]
        shirt_mask = np.isin(seg, shirt_classes)
        print(f"[DEBUG] Shirt mask pixel count: {np.sum(shirt_mask)}")

        # Optionally, restrict mask to pose-based ROI
        x1 = x2 = y1 = y2 = None
        try:
            REQUIRED = [11, 12, 23, 24]
            visible = all(landmarks[i].visibility > 0.5 for i in REQUIRED)
            if visible:
                coords = []
                for i in REQUIRED:
                    lm = landmarks[i]
                    x = int(np.clip(lm.x * w, 0, w - 1))
                    y = int(np.clip(lm.y * h, 0, h - 1))
                    coords.append((x, y))
                (ls_x, ls_y), (rs_x, rs_y), (lh_x, lh_y), (rh_x, rh_y) = coords
                x1 = max(0, min(ls_x, rs_x))
                x2 = min(w, max(ls_x, rs_x))
                torso_top = min(ls_y, rs_y)
                torso_bottom = max(lh_y, rh_y)
                y1 = int(torso_top + (torso_bottom - torso_top) * 0.25)
                y2 = int(torso_top + (torso_bottom - torso_top) * 0.75)
        except Exception:
            pass
        if x1 is None or x2 - x1 < 30 or y2 - y1 < 30:
            x1, x2, y1, y2 = self._fallback_chest_roi(h, w)

        # Mask out only the shirt region in the ROI
        # roi_mask = np.zeros_like(shirt_mask, dtype=bool)
        # roi_mask[y1:y2, x1:x2] = True
        # shirt_mask = shirt_mask & roi_mask
        # [DEBUG] Skipping ROI restriction for shirt mask
        print("[DEBUG] Skipping ROI restriction for shirt mask, using full shirt mask.")

        if np.sum(shirt_mask) < 10:
            print("[DEBUG] Shirt mask too small after ROI restriction.")
            return "unknown"

        # Get HSV pixels in shirt mask
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h_ch, s_ch, v_ch = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
        h_vals = h_ch[shirt_mask]
        s_vals = s_ch[shirt_mask]
        v_vals = v_ch[shirt_mask]

        # Filter out low-saturation/low-value pixels
        color_idx = (s_vals > 40) & (v_vals > 40)
        if np.sum(color_idx) < 10:
            print(f"[DEBUG] Not enough color pixels in mask. s>40 & v>40 count: {np.sum(color_idx)}")
            med_v = np.median(v_vals) if len(v_vals) else 0
            med_s = np.median(s_vals) if len(s_vals) else 0
            if med_v < 90:
                print("[DEBUG] Shirt detected as black (low value)")
                return "black"
            if med_v > 160:
                print("[DEBUG] Shirt detected as white (high value)")
                return "white"
            if med_s < 40:
                print("[DEBUG] Shirt detected as gray (low saturation)")
                return "gray"
            return "white"

        # Use 20th percentile for V to better detect dark shirts
        v_percentile = np.percentile(v_vals[color_idx], 20) if len(v_vals[color_idx]) else 0
        print(f"[DEBUG] V 20th percentile in mask: {v_percentile}")
        if v_percentile < 90:
            print("[DEBUG] Overriding to black due to low V 20th percentile (dark shirt)")
            return "black"
        
        # Print S and V percentiles for debugging
        s_percentile_20 = np.percentile(s_vals[color_idx], 20) if len(s_vals[color_idx]) else 0
        s_percentile_80 = np.percentile(s_vals[color_idx], 80) if len(s_vals[color_idx]) else 0
        v_percentile_80 = np.percentile(v_vals[color_idx], 80) if len(v_vals[color_idx]) else 0
        print(f"[DEBUG] S 20th percentile: {s_percentile_20}, S 80th percentile: {s_percentile_80}, V 80th percentile: {v_percentile_80}")
        # White shirt override: high V, low S
        if v_percentile_80 > 200 and s_percentile_80 < 60:
            print("[DEBUG] Overriding to white due to high V 80th percentile and low S 80th percentile (white shirt)")
            return "white"

        # Optionally, filter out very high V pixels (likely background/skin)
        filtered_idx = color_idx & (v_vals < 220)
        if np.sum(filtered_idx) >= 10:
            h_main = np.median(h_vals[filtered_idx])
            s_main = np.median(s_vals[filtered_idx])
            v_main = np.median(v_vals[filtered_idx])
            print(f"[DEBUG] HSV median in filtered mask: H={h_main}, S={s_main}, V={v_main}")
        else:
            h_main = np.median(h_vals[color_idx])
            s_main = np.median(s_vals[color_idx])
            v_main = np.median(v_vals[color_idx])
            print(f"[DEBUG] HSV median in mask: H={h_main}, S={s_main}, V={v_main}")

        # Map HSV to color name
        if s_main < 40:
            if v_main < 90:
                print("[DEBUG] Shirt detected as black (final mapping)")
                return "black"
            if v_main > 160:
                print("[DEBUG] Shirt detected as white (final mapping)")
                return "white"
            return "gray"
        if h_main < 10 or h_main > 170:
            print("[DEBUG] Shirt detected as red")
            return "red"
        elif h_main < 25:
            print("[DEBUG] Shirt detected as orange")
            return "orange"
        elif h_main < 35:
            print("[DEBUG] Shirt detected as yellow")
            return "yellow"
        elif h_main < 85:
            print("[DEBUG] Shirt detected as green")
            return "green"
        elif h_main < 130:
            print("[DEBUG] Shirt detected as blue")
            return "blue"
        elif h_main < 160:
            print("[DEBUG] Shirt detected as purple")
            return "purple"
        else:
            print("[DEBUG] Shirt detected as red (fallback)")
            return "red"
    
    @staticmethod
    def _detect_glasses(face_img):
        """Detect glasses using edge detection in the eye region."""
        if face_img is None or face_img.size == 0:
            return False
        
        h, w = face_img.shape[:2]
        
        eye_y1 = int(h * 0.2)
        eye_y2 = int(h * 0.5)
        eye_x1 = int(w * 0.1)
        eye_x2 = int(w * 0.9)
        
        eye_region = face_img[eye_y1:eye_y2, eye_x1:eye_x2]
        
        if eye_region.size == 0:
            return False
        
        gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        edge_density = np.sum(edges > 0) / edges.size
        
        sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        horizontal_edges = np.abs(sobel_x)
        horizontal_density = np.mean(horizontal_edges) / 255
        
        return bool(edge_density > 0.08 or horizontal_density > 0.15)
    
    @staticmethod
    def _get_age_group(age):
        """Convert age to age group."""
        if age < 20:
            return "<20"
        elif age < 30:
            return "20-30"
        else:
            return "30+"
    
    @staticmethod
    def _get_most_common(items):
        """Get the most common item from a list."""
        if not items:
            return None
        counter = Counter(items)
        return counter.most_common(1)[0][0]
    
    def _add_to_history(self, person_idx, traits):
        """Add traits to history buffer for a person."""
        if person_idx not in self.trait_history:
            self.trait_history[person_idx] = {
                "wearing_glasses": [],
                "gender": [],
                # "shirt_color": [],  # Removed shirt color
                "age": [],
                "emotion": []
            }
        
        for key, value in traits.items():
            if key in self.trait_history[person_idx] and value is not None:
                self.trait_history[person_idx][key].append(value)
    
    def _aggregate_traits(self):
        """Aggregate traits from history and return the most common values."""
        aggregated = []
        
        for person_idx, history in self.trait_history.items():
            person_data = {
                "wearing_glasses": self._get_most_common(history["wearing_glasses"]),
                "gender": self._get_most_common(history["gender"]),
                # "shirt_color": self._get_most_common(history["shirt_color"]),  # Removed shirt color
                "age": self._get_most_common(history["age"]),
                "emotion": self._get_most_common(history["emotion"])
            }
            # Only include if we have valid data
            if any(v is not None for v in person_data.values()):
                aggregated.append(person_data)
        
        return aggregated
    
    def _clear_history(self):
        """Clear the trait history buffer."""
        self.trait_history = {}
    
    def analyze_frame(self, frame):
        """
        Analyze a single frame and return structured output.
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            dict: Structured output with detection results
        """
        self.frame_count += 1
        h, w, _ = frame.shape
        
        # Convert BGR to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        # # Detect shirt color from pose
        # if self.frame_count % self.analyze_interval == 0:
        #     pose_results = self.pose_detector.detect(mp_image)
        #     if pose_results.pose_landmarks:
        #         landmarks = pose_results.pose_landmarks[0]
        #         shirt_color = self._detect_shirt_color(frame, landmarks, h, w)
        #         print(f"[DEBUG] Detected shirt color: {shirt_color}")
        #         if shirt_color:
        #             print(f"[DEBUG] Setting current_shirt_color to: {shirt_color}")
        #             self.current_shirt_color = shirt_color
        
        # Detect faces
        results = self.face_detector.detect(mp_image)
        
        outputs = []
        
        if results.detections:
            for i, detection in enumerate(results.detections):
                bbox = detection.bounding_box
                x, y = bbox.origin_x, bbox.origin_y
                bw, bh = bbox.width, bbox.height
                
                x = max(0, x)
                y = max(0, y)
                x2 = min(w, x + bw)
                y2 = min(h, y + bh)
                
                # Analyze face periodically
                if self.frame_count % self.analyze_interval == 0:
                    try:
                        face_img = frame[y:y2, x:x2]
                        if face_img.size > 0:
                            result = DeepFace.analyze(
                                face_img,
                                actions=["emotion", "gender", "age"],
                                enforce_detection=False,
                                silent=True
                            )
                            
                            emotion = result[0]["dominant_emotion"]
                            gender = result[0]["dominant_gender"]
                            age = result[0]["age"]
                            has_glasses = self._detect_glasses(face_img)
                            
                            self.current_face_data[i] = {
                                "wearing_glasses": has_glasses,
                                "gender": gender.lower(),
                                # "shirt_color": self.current_shirt_color,  # Removed shirt color
                                "age": self._get_age_group(age),
                                "emotion": emotion
                            }
                    except Exception:
                        pass
                
                # Add to history buffer
                if i in self.current_face_data:
                    self._add_to_history(i, self.current_face_data[i])
                
                # Get current data for this face (for display)
                if i in self.current_face_data:
                    output = self.current_face_data[i].copy()
                    output["bbox"] = {"x": x, "y": y, "width": bw, "height": bh}
                    outputs.append(output)
        
        # Aggregate and output every output_interval frames
        should_output = self.frame_count % self.output_interval == 0
        if should_output and self.trait_history:
            self.aggregated_output = {
                "persons": self._aggregate_traits(),
                "frame_count": self.frame_count,
                "aggregated_over_frames": self.output_interval
            }
            self._clear_history()
        
        return {
            "persons": outputs,
            "frame_count": self.frame_count,
            "aggregated_output": self.aggregated_output if should_output else None
        }
    
    def draw_annotations(self, frame, analysis_result):
        """
        Draw annotations on the frame based on analysis results.
        
        Args:
            frame: BGR image from OpenCV
            analysis_result: Result from analyze_frame()
            
        Returns:
            Annotated frame
        """
        h, w, _ = frame.shape
        
        for person in analysis_result.get("persons", []):
            bbox = person.get("bbox", {})
            x = bbox.get("x", 0)
            y = bbox.get("y", 0)
            bw = bbox.get("width", 0)
            bh = bbox.get("height", 0)
            x2 = x + bw
            y2 = y + bh
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
            
            # Display emotion on top
            emotion = person.get("emotion", "")
            if emotion:
                cv2.putText(frame, emotion, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display gender and age below the box
            gender = person.get("gender", "")
            age = person.get("age", "")
            if gender and age:
                label = f"{gender}, {age}"
                cv2.putText(frame, label, (x, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Display glasses
            wearing_glasses = person.get("wearing_glasses", False)
            glasses_text = "Glasses" if wearing_glasses else "No Glasses"
            cv2.putText(frame, glasses_text, (x, y2 + 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            
            # Display shirt color
            # shirt_color = person.get("shirt_color", "")
            # if shirt_color:
            #     cv2.putText(frame, f"Shirt: {shirt_color}", (x, y2 + 70),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        return frame
    
    def run_webcam(self, camera_id=0, show_window=True, print_json=False):
        """
        Run real-time analysis on webcam feed.
        
        Args:
            camera_id: Camera device ID (default: 0)
            show_window: Whether to show the video window (default: True)
            print_json: Whether to print JSON output each analysis (default: False)
        """
        cap = cv2.VideoCapture(camera_id)
        
        print("Press 'q' to quit, 'p' to print current JSON output")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Analyze frame
            result = self.analyze_frame(frame)
            
            # Print aggregated JSON output every output_interval frames (1 second)
            if print_json and result.get("aggregated_output"):
                print("\n" + "="*50)
                print(f"Aggregated Output (over {self.output_interval} frames):")
                print(json.dumps(result["aggregated_output"], indent=2))
                print("="*50)
            
            # Draw annotations
            if show_window:
                annotated_frame = self.draw_annotations(frame.copy(), result)
                cv2.imshow("Person Analyzer", annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("p"):
                # Print current JSON output
                print("\n" + "="*50)
                print("Current Detection Output:")
                print(json.dumps(result, indent=2))
                print("="*50 + "\n")
        
        cap.release()
        cv2.destroyAllWindows()
        
        return result


# Main entry point
if __name__ == "__main__":
    # analyze_interval=5: analyze every 5 frames (~6x per second)
    # output_interval=30: output aggregated result every 30 frames (~1 second)
    analyzer = PersonAnalyzer(analyze_interval=5, output_interval=30)
    analyzer.run_webcam(show_window=True, print_json=True)
