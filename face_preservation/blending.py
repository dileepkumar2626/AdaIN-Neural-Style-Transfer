import numpy as np
import cv2


def preserve_face(
    original_image: np.ndarray,
    stylized_image: np.ndarray,
    face_mask: np.ndarray,
    preserve_strength: float = 0.7,
    feather_radius: int = 15,
) -> np.ndarray:
    if original_image.shape != stylized_image.shape:
        raise ValueError("Original and stylized images must have identical dimensions.")
    if face_mask.shape != original_image.shape[:2]:
        raise ValueError("Face mask dimensions must match the image dimensions.")
    if not 0.0 <= preserve_strength <= 1.0:
        raise ValueError("preserve_strength must be between 0 and 1.")

    original = original_image.astype(np.float32)
    stylized = stylized_image.astype(np.float32)

    mask = (face_mask > 0).astype(np.float32)

    if feather_radius > 0:
        kernel_size = feather_radius * 2 + 1
        mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)

    alpha = mask * preserve_strength
    alpha = alpha[..., None]

    result = alpha * original + (1.0 - alpha) * stylized
    return np.clip(result, 0, 255).astype(np.uint8)
