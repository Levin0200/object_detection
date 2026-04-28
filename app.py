import io
from collections import Counter
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO


TARGET_MAX_SIDE = 960


@st.cache_resource
def load_model(model_path: str = "yolov8n.pt") -> YOLO:
    """Load and cache the YOLOv8 model for CPU inference."""
    return YOLO(model_path)


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Convert bytes to RGB image and resize while preserving aspect ratio."""
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = np.array(pil_image)

    h, w = image.shape[:2]
    max_side = max(h, w)
    if max_side > TARGET_MAX_SIDE:
        scale = TARGET_MAX_SIDE / max_side
        new_w, new_h = int(w * scale), int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return image


def run_detection(
    model: YOLO, image_rgb: np.ndarray, conf_threshold: float = 0.25
) -> Tuple[np.ndarray, List[Dict[str, float]], Counter]:
    """Run YOLOv8 detection and return raw boxes, object list, and counts."""
    results = model.predict(
        source=image_rgb,
        conf=conf_threshold,
        device="cpu",
        verbose=False,
    )

    result = results[0]
    detections: List[Dict[str, float]] = []
    object_counter: Counter = Counter()

    if result.boxes is not None and len(result.boxes) > 0:
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()

        for box, cls_idx, conf in zip(boxes_xyxy, classes, confidences):
            label = model.names.get(int(cls_idx), str(cls_idx))
            detections.append(
                {
                    "label": label,
                    "confidence": float(conf),
                    "x1": float(box[0]),
                    "y1": float(box[1]),
                    "x2": float(box[2]),
                    "y2": float(box[3]),
                }
            )
            object_counter[label] += 1

    return image_rgb, detections, object_counter


def draw_boxes(image_rgb: np.ndarray, detections: List[Dict[str, float]]) -> np.ndarray:
    """Draw bounding boxes and labels on a copy of the input image."""
    image_bgr = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)

    for det in detections:
        x1, y1 = int(det["x1"]), int(det["y1"])
        x2, y2 = int(det["x2"]), int(det["y2"])
        label_text = f'{det["label"]} {det["confidence"]:.2f}'

        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

        (text_w, text_h), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        text_y = max(y1 - 8, text_h + 4)
        cv2.rectangle(
            image_bgr,
            (x1, text_y - text_h - baseline - 4),
            (x1 + text_w + 6, text_y + 2),
            (0, 255, 0),
            -1,
        )
        cv2.putText(
            image_bgr,
            label_text,
            (x1 + 3, text_y - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def display_results(
    annotated_image: np.ndarray, detections: List[Dict[str, float]], object_counter: Counter
) -> None:
    """Render visual and tabular outputs in Streamlit."""
    st.subheader("Detection Results")
    st.image(annotated_image, caption="Detected objects", use_container_width=True)

    total_objects = sum(object_counter.values())
    st.markdown(f"**Total objects detected:** {total_objects}")

    if total_objects == 0:
        st.info("No objects detected in this image.")
        return

    st.markdown("### Object Count Statistics")
    stats_df = pd.DataFrame(
        [{"Object": label, "Count": count} for label, count in object_counter.items()]
    ).sort_values(by="Count", ascending=False)
    st.dataframe(stats_df, use_container_width=True, hide_index=True)

    st.markdown("### Object List (Label + Confidence)")
    det_df = pd.DataFrame(
        [{"Label": d["label"], "Confidence": round(d["confidence"], 3)} for d in detections]
    )
    st.dataframe(det_df, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="Mobile Object Detection App", layout="centered")
    st.title("Mobile Object Detection App")
    st.write(
        "Capture an image from your smartphone camera or upload one, then run YOLOv8 "
        "object detection with bounding boxes, labels, confidence scores, and counts."
    )

    st.markdown("## Input Section")
    captured_file = st.camera_input("Take a photo")
    uploaded_file = st.file_uploader(
        "Or upload an image", type=["jpg", "jpeg", "png", "webp"]
    )

    selected_file = captured_file if captured_file is not None else uploaded_file

    if selected_file is None:
        st.info("Please capture or upload an image to start detection.")
        return

    image_bytes = selected_file.getvalue()
    image_rgb = preprocess_image(image_bytes)
    st.image(image_rgb, caption="Input image", use_container_width=True)

    conf_threshold = st.slider(
        "Confidence threshold", min_value=0.1, max_value=0.9, value=0.25, step=0.05
    )

    if st.button("Run Detection", type="primary"):
        with st.spinner("Loading model and running inference on CPU..."):
            model = load_model()
            image_rgb_processed, detections, object_counter = run_detection(
                model=model,
                image_rgb=image_rgb,
                conf_threshold=conf_threshold,
            )
            annotated = draw_boxes(image_rgb_processed, detections)

        display_results(annotated, detections, object_counter)

        st.markdown("## Extensions (Future Work)")
        st.markdown(
            "- AI explanation for detected objects\n"
            "- Dangerous object alerting\n"
            "- Real-time video detection\n"
            "- Jetson-based edge deployment"
        )


if __name__ == "__main__":
    main()
