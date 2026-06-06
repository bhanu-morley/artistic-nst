"""
Neural Style Transfer - Gradio UI
User-facing application that:
    Accepts Content + Style images
    Runs NST model to generate artwork
    Saves all images (original + stylized) to MongoDB/GridFS async
    Provides download + live preview
"""

# -------------------------------------------------------------------------
# 1. Suppress TensorFlow / Keras Warnings for Clean Console Output
# -------------------------------------------------------------------------

import io, os, warnings, logging
import threading
import tempfile
# from datetime import datetime
# from zoneinfo import ZoneInfo

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'        #TF minimal logging
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
logger = logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("tensorflow.keras").setLevel(logging.ERROR)
logging.getLogger("tf_keras").setLevel(logging.ERROR)


# -------------------------------------------------------------------------
# 2. Core Imports
# -------------------------------------------------------------------------

import gradio as gr
import numpy as np
from PIL import Image
import uuid
from dotenv import load_dotenv


# -------------------------------------------------------------------------
# 3. Load Env (MongoDB Credentials)
# -------------------------------------------------------------------------

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "artistic_nst")


# -------------------------------------------------------------------------
# 4. Import Model Loader + Style Transfer Function
# -------------------------------------------------------------------------

from api import load_nst_model, transfer_style


# -------------------------------------------------------------------------
# 5. Import database abstraction for GridFS
# -------------------------------------------------------------------------

from src.database.db import NSTDatabase


# -------------------------------------------------------------------------
# 6. Initialize Database (Connect to MongoDB)
# -------------------------------------------------------------------------

db = NSTDatabase(MONGO_URI, DB_NAME)


# -------------------------------------------------------------------------
# 7. Load Style Transfer Model (once at startup)
#    Model directory should contain saved Keras weights / model config
# -------------------------------------------------------------------------

MODEL_DIR = "src/trained_models"
load_nst_model(MODEL_DIR)


# -------------------------------------------------------------------------
# 8. Async DB Save Worker
# Saves images in a background thread so UI remains fast
# -------------------------------------------------------------------------

def save_to_db_async(content, style, output):
    try:
        db.save_triplet(content, style, output)
    except Exception as e:
        print(f"[DB WARN] {e}")


# -------------------------------------------------------------------------
# 9. Neural Style Transfer Pipeline
#    Accept PIL → convert to numpy → run model → convert to PIL → save temp
# -------------------------------------------------------------------------

def run_style_transfer(content, style):
    if content is None or style is None:
        return None, None

    # Model inference: Blend style into content
    result = transfer_style(np.array(content), np.array(style), MODEL_DIR)

    # Output comes as normalized float → convert to uint8
    result = (result * 255).astype(np.uint8)
    out_img = Image.fromarray(result)

    # Save output as temp file — user can download
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    out_img.save(tmp.name, format="PNG")

    # Save all 3 images to DB in the background (non-blocking)
    threading.Thread(target=save_to_db_async, args=(content, style, out_img), daemon=True).start()

    # Return PIL image + file path for Gradio Download element
    return out_img, tmp.name


# -------------------------------------------------------------------------
# 10. Custom UI Theme / Styles
# -------------------------------------------------------------------------

CSS = """
/* Main Title */
.app-title, .app-title * {
    font-size: 2.2rem !important;
    font-weight: 700 !important;
    text-align: center !important;
    color: white !important;
}

.gradio-container h1.app-title,
.gradio-container .app-title h1 {
    color: white !important;
}

.app-title:hover, .app-title:hover * {
    color: #4CFF4C !important;
    cursor: pointer;
}


/* Section Headers */
.header-title {
    font-size: 1.13rem !important;
    font-weight: 500 !important;
    text-align: center !important;
    color: white !important;
    letter-spacing: 0.5px !important;
    text-transform: uppercase !important;
    opacity: 0.7 !important;
}


/* Subtitle under tilte */
#subtitle, #subtitle * {
    text-align: center !important;
    display: block !important;
    margin: 0 auto !important;
    font-size: 1.2rem !important;
    opacity: 0.9 !important;
    color: grey !important;
}

#subtitle:hover, #subtitle:hover * {
    cursor: pointer !important;
    color:  white !important;
}


/* Global Theme */
:root, body, .gradio-container {
    background-color:#000 !important;
    color:#fff !important;
    font-family: 'Inter', sans-serif !important;
    font-weight:600 !important;
}


/* Upload container section */
.upload-section {
    padding-bottom: 45px !important;
    border: 3px dotted #666 !important;
    border-radius: 18px !important;
    width: 80% !important;
    margin: 0 auto !important;
}


/* Upload boxes */
.upload-box, .gradio-image, .gradio-container .image-preview {
    padding: -50px !important;
    background:#111 !important;
    border:2px dashed #666 !important;
    border-radius:14px !important;
    width:450px !important;
    height:350px !important;
    margin:auto !important;
    font-weight:600 !important;
    trasnition: border-color 0.2s ease-in-out !important;
}

.upload-box:hover { border-color:white !important; }

.gradio-image img {
    border-radius:12px !important;
}

.gradio-container input[type="file"] {
    color:white !important;
}

.output-image, .gradio-container .image-preview {
    background:#111 !important;
    border:2px solid #555 !important;
    border-radius:12px !important;
}


/* Start Button */
#start-btn {
    background: #0A84FF !important;
    border-color: #0A84FF !important;
    color: white !important;
    width: 320px !important;
    height: 60px !important;
    font-size: 18px !important;
    font-weight: 700 !important;
    margin: 0 auto !important;
    display: block !important;
    margin-top: 20px !important;
}

#start-btn:hover {
    background: #4CFF4C !important;
    border-color: #4CFF4C !important;
    color: black !important;
}

button, .gr-button {
    background:none !important;
    border:none !important;
}

footer, .gradio-container footer {
    display:none !important;
}

input, textarea, select {
    background:#111 !important;
    color:#fff !important;
    font-weight:600 !important;
    border:1px solid #444 !important;
}

* {
    box-shadow:none !important;
}


/* RESPONSIVE DESIGN FOR MOBILE */
@media (max-width: 768px) {

    .app-title, .app-title * {
        font-size: 1.6rem !important;
    }

    #subtitle, #subtitle * {
        font-size: 1rem !important;
    }

    .upload-section {
        padding: 15px !important;
        padding-bottom: 50px !important;
        width: 100% !important;
    }

    .upload-box, .gradio-image, .gradio-container .image-preview {
        width: 92% !important;
        height: 260px !important;
    }

    #start-btn {
        width: 70% !important;
        height: 50px !important;
        font-size: 16px !important;
    }
}


/* Tablet or Small laptop view (between PC & Mobile) */
@media (min-width: 768px) and (max-width: 1180px) {

    .app-title, .app-title * {
        font-size: 1.9rem !important;
    }

    #subtitle, #subtitle * {
        font-size: 1.08rem !important;
    }

    .upload-section {
        padding: 25px !important;
        width: 96% !important;
        margin: 0 auto !important;
        padding-bottom:50px !important;
    }

    .upload-box, .gradio-image, .gradio-container .image-preview {
        width: 90% !important;
        height: 90% !important;
    }

    #start-btn {
        width: 260px !important;
        height: 55px !important;
        font-size: 17px !important;
        margin-top: 18px !important;
    }
}

"""

# -------------------------------------------------------------------------
# 11. UI Definition (Gradio Blocks Layout)
# -------------------------------------------------------------------------

with gr.Blocks(theme=gr.themes.Soft(primary_hue="gray"), title="Neural Style Transfer", css=CSS) as app:

    # Header
    gr.HTML("<h1 class='app-title'><b>Neural Style Transfer</b></h1>")
    gr.Markdown("Upload Content + Style images → AI blends them into Art", elem_id="subtitle")


    # Upload area — side by side images
    with gr.Row(elem_classes="upload-section"):

        with gr.Column():
            gr.HTML("<div class='header-title'>Content Image</div>")
            content_uploader = gr.Image(
                type="pil",
                sources=["upload"],
                image_mode="RGB",
                elem_classes="upload-box"
            )

        with gr.Column():
            gr.HTML("<div class='header-title'>Style Image</div>")
            style_uploader = gr.Image(
                type="pil",
                sources=["upload"],
                image_mode="RGB",
                elem_classes="upload-box"
            )


    # Output preview + download button (hidden until ready)
    output_img = gr.Image(label="Styled Result", width=450, height=520, visible=False)
    download_btn = gr.File(label="Download Result", visible=False)


    # Action + loading message
    loading = gr.Markdown("⏳ Processing... Please wait", visible=False)
    start_btn = gr.Button("Start Style Transfer", elem_id="start-btn")
    

    # ---------------------------------------------------------------------
    # 12. UI Callback: Run style transfer, update results
    # ---------------------------------------------------------------------

    def show_result(content, style):
        
        if content is None or style is None:
            raise gr.Error("Please upload both Content and Style images before starting!")
        
        img, file_obj = run_style_transfer(content, style)
        return gr.update(value=img, visible=True), gr.update(value=file_obj, visible=True)
    
    
    # Click flow:
    # 1) show spinner
    # 2) run inference
    # 3) hide spinner, show result
    start_btn.click(
        show_result,
        [content_uploader, style_uploader],
        [output_img, download_btn],
        queue=True
    ).then(
        lambda: gr.update(visible=False),
        None,
        [loading]
    )

    start_btn.click(
        lambda: gr.update(visible=True),
        None,
        [loading]
    )

    # ---------------------------------------------------------------------
    # 13. Clearing images resets output view
    # ---------------------------------------------------------------------

    def clear_output(_=None):
        return (
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False, file=None)
        )

    content_uploader.clear(fn=clear_output, outputs=[output_img, download_btn])
    style_uploader.clear(fn=clear_output, outputs=[output_img, download_btn])


# -------------------------------------------------------------------------
# 14. Temp File Cleanup (auto-delete generated images after 3 hours)
# -------------------------------------------------------------------------

def cleanup_temp_files(hours=3):
    import time
    now = time.time()
    tmpdir = tempfile.gettempdir()
    for f in os.listdir(tmpdir):
        if f.endswith(".png"):
            fp = os.path.join(tmpdir, f)
            if os.path.isfile(fp) and now - os.path.getmtime(fp) > hours*3600:
                try:
                    os.remove(fp)
                except:
                    pass

# Run cleanup in background thread
threading.Thread(target=cleanup_temp_files, daemon=True).start()




app.launch(server_port=7860, share=False)