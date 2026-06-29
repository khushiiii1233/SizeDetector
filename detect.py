import cv2
import numpy as np
from ultralytics import YOLO
from pupil_apriltags import Detector

tag_size_cm = 4.5
camera_matrix = np.load("camera_matrix.npy")
dist_coeffs = np.load("dist_coeffs.npy")

detector = Detector(
    families="tag36h11"
)

yolo = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

object_points = np.array([
    [-tag_size_cm/2, -tag_size_cm/2, 0],
    [ tag_size_cm/2, -tag_size_cm/2, 0],
    [ tag_size_cm/2,  tag_size_cm/2, 0],
    [-tag_size_cm/2,  tag_size_cm/2, 0]
], dtype=np.float32)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    tags = detector.detect(gray)

    results = yolo(frame)

    pixels_per_cm = None


    for tag in tags:

        corners = tag.corners
        image_points = corners.astype(np.float32)

        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs
        )
        distance_cm = np.linalg.norm(tvec)

        cv2.putText(
            frame,
            f"Distance: {distance_cm:.1f} cm",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )
        corners_int = corners.astype(int)

        for i in range(4):

            pt1 = tuple(corners_int[i])
            pt2 = tuple(corners_int[(i + 1) % 4])

            cv2.line(
                frame,
                pt1,
                pt2,
                (0, 255, 0),
                2
            )

        for corner in corners:

            cv2.circle(
                frame,
                (int(corner[0]), int(corner[1])),
                5,
                (0, 0, 255),
                -1
            )

        pixel_width = np.linalg.norm(
            corners[0] - corners[1]
        )

        pixels_per_cm = pixel_width / tag_size_cm

        center_x = int(tag.center[0])
        center_y = int(tag.center[1])

        cv2.putText(
            frame,
            f"Scale: {pixels_per_cm:.2f} px/cm",
            (center_x - 80, center_y - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

    if pixels_per_cm is not None:

        for result in results:

            for detection in result.boxes:

                x1, y1, x2, y2 = detection.xyxy[0]

                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)

                width_pixels = x2 - x1
                height_pixels = y2 - y1

                width_cm = width_pixels / pixels_per_cm
                height_cm = height_pixels / pixels_per_cm

                class_id = int(detection.cls[0])
                confidence = float(detection.conf[0])

                class_name = result.names[class_id]

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"{class_name} {confidence:.2f}",
                    (x1, y1 - 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"{width_cm:.1f}cm x {height_cm:.1f}cm",
                    (x1, y1 - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )

    cv2.imshow(
        "AprilTag + YOLO Measurement",
        frame
    )

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()