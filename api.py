"""
Neural Style Transfer API Module
--------------------------------
Handles:
- Loading TensorFlow-Hub NST model only once (cached)
- Preprocessing content & style images
- Running style transfer
- Returning stylized output (NumPy image)
"""
import os, warnings, logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
logger = logging.getLogger("tensorflow")
logger.setLevel(logging.ERROR)
logging.getLogger("tensorflow.keras").setLevel(logging.ERROR)
logging.getLogger("tf_keras").setLevel(logging.ERROR)
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from functools import lru_cache


# ---------------------------------------------------------
# 1. Optimize TensorFlow CPU Execution
# ---------------------------------------------------------
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.set_soft_device_placement(True)


# ---------------------------------------------------------
# 2. Load Model ONCE & Cache It (Important for Gradio/Web)
# ---------------------------------------------------------
@lru_cache(maxsize=1)
def load_nst_model(model_path: str):
    """
    Load style transfer model from TensorFlow Hub and cache it.

    Why caching?
    - Avoids model reloading for every request → faster UI
    - Only loads once even if function is called again

    :param model_path: Path to TF-Hub saved model folder
    :return: Loaded model
    """

    tf.get_logger().setLevel(logging.ERROR)
    logging.getLogger("tensorflow_hub").setLevel(logging.ERROR)

    print("Loading Style Transfer Model...")
    model = hub.load(model_path)
    print("Model Loaded & Cached")
    return model


# ---------------------------------------------------------
# 3. Preprocess Image
# ---------------------------------------------------------
def preprocess_image(img: np.ndarray, max_dim: int = 1024):
    """
    Prepare image for TensorFlow model:

    Steps:
        1. Make sure image has 3 channels (RGB)
        2. Resize if image is too large (max: 1024px)
        3. Convert uint8 → float32 & normalize to [0,1]
        4. Add batch dimension → (1, H, W, 3)

    :param img: Raw image array (H, W, C)
    :param max_dim: Max allowed dimension for speed
    :return: Preprocessed tensor ready for model (1,H,W,3)
    """

    # Remove alpha channel (RGBA → RGB)
    img = img[..., :3]

    # Resize large images to avoid slow inference / OOM
    h, w = img.shape[:2]
    scale = max_dim / max(h, w)
    if scale < 1:  
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h))

    # Normalize to [0,1]
    img = img.astype(np.float32) / 255.0

    # Add batch dim → model expects 4D input
    return img[np.newaxis, ...]


# ---------------------------------------------------------
# 4. Apply Style Transfer
# ---------------------------------------------------------
def transfer_style(content_img: np.ndarray, style_img: np.ndarray, model_path: str):
    """
    Perform Artistic Neural Style Transfer using pre-trained TF-Hub model.

    Pipeline:
        1. Preprocess content image
        2. Preprocess style image
        3. Downscale style (256x256) → faster & sharper style extraction
        4. Load cached model
        5. Run inference
        6. Remove batch dim + return stylized RGB image

    :param content_img: NumPy content image (H,W,3)
    :param style_img: NumPy style image (H,W,3)
    :param model_path: Path to saved TF-Hub model
    :return: Stylized result (H,W,3 NumPy array, float32 0-1)
    """

    print("Preprocessing images...")
    content_tensor = preprocess_image(content_img)
    style_tensor = preprocess_image(style_img)

    # Resize style smaller for speed & better texture extraction
    style_tensor = tf.image.resize(style_tensor, (256, 256))

    # Load cached model
    model = load_nst_model(model_path)

    print("Running Style Transfer...")
    outputs = model(tf.constant(content_tensor), tf.constant(style_tensor))

    # Remove batch dimension: (1,H,W,3) → (H,W,3)
    result = outputs[0].numpy().squeeze(axis=0)

    print("Style Transfer Completed")
    return result
