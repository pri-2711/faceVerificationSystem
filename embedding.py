# =============================================================
#  embedding.py  —  Face detection + embedding generation
#
#  Uses:
#    MTCNN             → detect and align faces
#    InceptionResnetV1 → generate 512-dim embedding vector
#
#  Both models are loaded once at import time and reused
#  everywhere else in the project.
# =============================================================

import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import config

# Load models once — shared across the whole project
mtcnn = MTCNN(
    image_size  = config.IMAGE_SIZE,
    device      = config.DEVICE,
    keep_all    = True,      # detect all faces in frame, not just largest
    post_process= True       # normalise output for InceptionResnetV1
)

facenet = InceptionResnetV1(pretrained="vggface2").eval().to(config.DEVICE)


# -------------------------------------------------------------

def _to_pil(image):
    """Convert numpy BGR (OpenCV) or PIL image to RGB PIL image."""
    if isinstance(image, np.ndarray):
        return Image.fromarray(image[:, :, ::-1])   # BGR → RGB
    return image                                     # already PIL


def get_single_embedding(image):
    """
    Detect the first (largest) face in the image and return its embedding.

    Args:
        image : PIL Image or numpy BGR array

    Returns:
        embedding (np.ndarray, shape 512) — or None if no face found
    """
    pil = _to_pil(image)
    faces = mtcnn(pil)          # tensor of aligned face crops, or None

    if faces is None:
        return None

    face_tensor = faces[0].unsqueeze(0).to(config.DEVICE)

    with torch.no_grad():
        emb = facenet(face_tensor)

    return emb.squeeze().cpu().numpy()


def get_all_embeddings(image):
    """
    Detect ALL faces in an image and return an embedding for each.
    Used during real-time webcam verification and group photo verification.

    Args:
        image : PIL Image or numpy BGR array

    Returns:
        embeddings (list of np.ndarray)   — one per detected face
        boxes      (list of [x1,y1,x2,y2])— bounding boxes, same order
    """
    pil = _to_pil(image)

    boxes, _ = mtcnn.detect(pil)
    faces    = mtcnn(pil)

    if faces is None or boxes is None:
        return [], []

    embeddings = []
    with torch.no_grad():
        for face in faces:
            emb = facenet(face.unsqueeze(0).to(config.DEVICE))
            embeddings.append(emb.squeeze().cpu().numpy())

    return embeddings, boxes.tolist()