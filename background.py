import cv2
import numpy as np
import requests

cap = cv2.VideoCapture("/dev/video0")
# configure camera for 480p @ 30 FPS (DroidCam)
width, height, fps = 640, 480, 30
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv2.CAP_PROP_FPS, fps)


def get_mask(frame, bodypix="http://localhost:9000"):
    _, data = cv2.imencode(".jpg", frame)
    headers = {"Content-Type": "application/octet-stream"}
    r = requests.post(bodypix, data.tobytes(), headers=headers)

    # conver raw bytes to a numpy array
    # raw data is an uint8[width * height] array with values 0 or 1
    mask = np.frombuffer(r.content, dtype=np.uint8).reshape(
        (frame.shape[0], frame.shape[1])
    )
    return mask


def post_process_mask(mask):
    mask = cv2.dilate(mask, np.ones((10, 10), np.uint8), iterations=1)
    mask = cv2.blur(mask.astype(float), (30, 30))
    return mask


# read in a "virtual background" (should be in 16:9 aspect ratio)
replacement_bg_raw = cv2.imread("mfbackground.jpg")

# resize to match the frame (width and height from before)
replacement_bg = cv2.resize(replacement_bg_raw, (width, height))

success, frame = cap.read()
if success:
    mask = post_process_mask(get_mask(frame))

    # combine the background and foreground, using the mask and its inverse
    inv_mask = 1 - mask
    for c in range(frame.shape[2]):
        frame[:, :, c] = frame[:, :, c] * mask + replacement_bg[:, :, c] * inv_mask

    cv2.imwrite("test.jpg", frame)
