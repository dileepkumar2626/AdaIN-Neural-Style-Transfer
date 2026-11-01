import numpy as np
import cv2


def get_face_mask(image_rgb: np.ndarray) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    try:
        import mediapipe as mp
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, refine_landmarks=True
        )
        results = face_mesh.process(image_rgb)
        face_mesh.close()
        if results.multi_face_landmarks:
            FACE_OVAL = [
                10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361,
                288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149,
                150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
            ]
            lm = results.multi_face_landmarks[0].landmark
            pts = np.array(
                [[int(lm[i].x * w), int(lm[i].y * h)] for i in FACE_OVAL],
                dtype=np.int32,
            )
            cv2.fillPoly(mask, [pts], 255)
            return mask
    except Exception:
        pass

    try:
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        for x, y, fw, fh in faces:
            cx, cy = x + fw // 2, y + fh // 2
            cv2.ellipse(mask, (cx, cy), (int(fw * 0.5), int(fh * 0.6)), 0, 0, 360, 255, -1)
            break
    except Exception:
        pass

    return mask
