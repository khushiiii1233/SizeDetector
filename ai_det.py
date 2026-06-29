import cv2
import numpy as np
from ultralytics import YOLO
from transformers import pipeline
from PIL import Image

camera_matrix = np.load("camera_matrix.npy")
dist_coeffs = np.load("dist_coeffs.npy")

depth_scale = 0.35

depth_estimator = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf"
)

yolo = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame.")
        break

    frame = cv2.undistort(
        frame,
        camera_matrix,
        dist_coeffs
    )

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    pil_image = Image.fromarray(rgb)

    depth = depth_estimator(pil_image)

    depth_map = np.array(
        depth["depth"]
    )

    results = yolo(frame)

    for result in results:

        for detection in result.boxes:

            x1, y1, x2, y2 = detection.xyxy[0]

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            width_pixels = x2 - x1
            height_pixels = y2 - y1

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            if (
                0 <= cy < depth_map.shape[0]
                and
                0 <= cx < depth_map.shape[1]
            ):

                depth_value = float(
                    depth_map[cy, cx]
                )

            else:

                depth_value = 0

            distance_cm = (
                depth_value *
                depth_scale
            )

            width_cm = (
                width_pixels *
                distance_cm /
                1000
            )

            height_cm = (
                height_pixels *
                distance_cm /
                1000
            )

            angle = np.degrees(
                np.arctan2(
                    (y2 - y1),
                    (x2 - x1)
                )
            )

            class_id = int(
                detection.cls[0]
            )

            confidence = float(
                detection.conf[0]
            )

            class_name = result.names[
                class_id
            ]

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.circle(
                frame,
                (cx, cy),
                4,
                (0, 0, 255),
                -1
            )

            cv2.putText(
                frame,
                f"{class_name} {confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Depth:{depth_value:.1f}",
                (x1, y1 - 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Dist:{distance_cm:.1f}cm",
                (x1, y1 - 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 255),
                2
            )

            cv2.putText(
                frame,
                f"{width_cm:.1f} x {height_cm:.1f} cm",
                (x1, y1 - 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Angle:{angle:.1f}",
                (x1, y1 - 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 200, 0),
                2
            )

    depth_display = cv2.normalize(
        depth_map,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    depth_display = depth_display.astype(
        np.uint8
    )

    depth_colormap = cv2.applyColorMap(
        depth_display,
        cv2.COLORMAP_INFERNO
    )

    cv2.imshow(
        "Camera Feed",
        frame
    )

    cv2.imshow(
        "Depth Map",
        depth_colormap
    )

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()