import cv2
import numpy as np
import glob

# Number of INNER corners
chessboard_size = (7, 6)

# Real size of one square (cm)
square_size = 2.5

# Arrays for calibration
object_points = []
image_points = []

# Create chessboard world coordinates
objp = np.zeros(
    (chessboard_size[0] * chessboard_size[1], 3),
    np.float32
)

objp[:, :2] = np.mgrid[
    0:chessboard_size[0],
    0:chessboard_size[1]
].T.reshape(-1, 2)

objp *= square_size

# Load images
images = glob.glob("calibration_images/*.jpg")

print(f"Found {len(images)} images")

for fname in images:

    img = cv2.imread(fname)

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    found, corners = cv2.findChessboardCorners(
        gray,
        chessboard_size,
        None
    )

    if found:

        object_points.append(objp)
        image_points.append(corners)

        cv2.drawChessboardCorners(
            img,
            chessboard_size,
            corners,
            found
        )

        cv2.imshow(
            "Corners",
            img
        )

        cv2.waitKey(200)

cv2.destroyAllWindows()

# Calibrate camera
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    object_points,
    image_points,
    gray.shape[::-1],
    None,
    None
)

print("\nCamera Matrix:\n")
print(camera_matrix)

print("\nDistortion Coefficients:\n")
print(dist_coeffs)

# Save files
np.save(
    "camera_matrix.npy",
    camera_matrix
)

np.save(
    "dist_coeffs.npy",
    dist_coeffs
)

print("\nCalibration complete!")
print("camera_matrix.npy saved")
print("dist_coeffs.npy saved")