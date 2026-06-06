# Admin Dashboard (thumbnail-on-the-fly, cached)

import os, io, base64, warnings, logging, uuid, functools
from PIL import Image
import gradio as gr
from dotenv import load_dotenv
from src.database.db import NSTDatabase

# ---------------------------------------------------------------------------
# 1. Suppress TensorFlow & Misc Warnings (to keep admin UI console clean)
# ---------------------------------------------------------------------------

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'         # Silence TF logs
warnings.filterwarnings("ignore")                # Disable Python warnings
logging.getLogger("tensorflow").setLevel(logging.ERROR)


# ---------------------------------------------------------------------------
# 2. Load Environment Variables (Mongo credentials)
# ---------------------------------------------------------------------------

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "artistic_nst")


# ---------------------------------------------------------------------------
# 3. Connect to MongoDB (Fail-fast if unreachable)
# ---------------------------------------------------------------------------

try:
    db = NSTDatabase(MONGO_URI, DB_NAME)
    print("MongoDB connected")
except Exception as e:
    print("MongoDB connection failed:", e)
    print("Start MongoDB and retry.")
    raise SystemExit(1)


# ---------------------------------------------------------------------------
# 4. Config: Pagination + active session tokens
# ---------------------------------------------------------------------------

PAGE_SIZE = 10                         # Records per admin page
valid_tokens = set()                   # Track logged-in admin sessions


# ---------------------------------------------------------------------------
# 5. Cached thumbnail generator (base64) for fast pagination display
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=512)
def _thumb_b64_cached(file_id: str, max_side: int = 256) -> str:
    """
    Load raw image bytes from GridFS, downscale to thumbnail,
    convert to Base64 → return as <img> data URL.
    Cached by file_id to avoid recompressing every page click.
    """
    data = db.get_file_bytes(file_id)
    img = Image.open(io.BytesIO(data)).convert("RGBA")

    # Create proportional thumbnail
    img.thumbnail((max_side, max_side))

    # Convert to PNG bytes
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)

    # Encode for HTML <img>
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


# ---------------------------------------------------------------------------
# 6. Render image cell (thumbnail + download link + lightbox viewer)
# ---------------------------------------------------------------------------

def img_cell(file_id: str, download_name: str) -> str:
    b64url = _thumb_b64_cached(file_id, max_side=256)
    return f"""
    <img src="{b64url}" width="120" style="border:2px solid #777;cursor:pointer"
         onclick="document.getElementById('lightbox-img').src='{b64url}';
                  document.getElementById('lightbox').style.display='flex';">
    <br><a href="{b64url}" download="{download_name}" style="color:#0af">Download</a>
    """


# ---------------------------------------------------------------------------
# 7. Single HTML table row (content / style / output / timestamp)
# ---------------------------------------------------------------------------

def render_row(rec: dict) -> str:
    # Backward compatible — fall back to original file IDs if no thumb
    content_id = rec.get("content_thumb_id") or rec["content_file_id"]
    style_id   = rec.get("style_thumb_id")   or rec["style_file_id"]
    output_id  = rec.get("output_thumb_id")  or rec["output_file_id"]

    ts = rec["created_at"]      # Already formatted IST string from DB

    return f"""
    <tr style='border:1px solid #555'>
        <td>{img_cell(content_id, "content.png")}</td>
        <td>{img_cell(style_id,   "style.png")}</td>
        <td>{img_cell(output_id,  "output.png")}</td>
        <td style="white-space:nowrap;">{ts}</td>
    </tr>
    """


# ---------------------------------------------------------------------------
# 8. Render page of results (HTML table)
# ---------------------------------------------------------------------------

def load_page(page: int) -> str:
    skip = max(page * PAGE_SIZE, 0)
    rows = db.list_records(skip=skip, limit=PAGE_SIZE)

    # Styling + table head
    html = """
    <style>
    table {width:100%;color:white;border-collapse:collapse;font-size:14px;}
    th,td {padding:8px;border:1px solid #444;}
    th {background:#222;}
    </style>
    <table>
        <tr><th>Content</th><th>Style</th><th>Output</th><th>Timestamp</th></tr>
    """
    for r in rows:
        html += render_row(r)

    html += "</table>"
    return html


# ---------------------------------------------------------------------------
# 9. Auth + State handlers
# ---------------------------------------------------------------------------

def do_login(u, p):
    """
    Validate admin user. If success:
     - create session token
     - reveal dashboard controls
    Else: show login error
    """
    if db.verify_admin(u, p):
        token = uuid.uuid4().hex
        valid_tokens.add(token)

        return (
            token,                                 # Save token
            gr.update(visible=False),              # hide login msg
            gr.update(value="<p style='text-align:center;color:#ffd54f'>Logged in - click Refresh to load records</p>", visible=True),
            gr.update(visible=True),               # show refresh
            gr.update(visible=True),               # show logout
            gr.update(visible=True),               # show prev
            gr.update(visible=True),               # show next
            token                                  # state token
        )

    # Login failed
    return (
        "",                                       # token reset
        gr.update(value="Invalid credentials", visible=True),
        gr.update(visible=False),                 # hide table
        gr.update(visible=False),                 # hide refresh
        gr.update(visible=False),                 # hide logout
        gr.update(visible=False),                 # hide prev
        gr.update(visible=False),                 # hide next
        ""
    )

def refresh_logs(token, page):
    if token not in valid_tokens:
        return "Session expired. Login again."
    return load_page(int(page or 0))

def logout(token):
    """
    End session — clear token + reset UI to login state
    """
    valid_tokens.discard(token)
    return (
        "",                          # clear token
        gr.update(visible=True),     # show login msg again
        gr.update(visible=False),    # hide table
        gr.update(visible=False),    # hide refresh
        gr.update(visible=False),    # hide logout
        gr.update(visible=False),    # hide prev
        gr.update(visible=False),    # hide next
        0                            # reset page
    )

def next_page(token, page):
    if token not in valid_tokens:
        return 0, "Session expired. Login again."
    new_p = int(page or 0) + 1
    return new_p, load_page(new_p)

def prev_page(token, page):
    if token not in valid_tokens:
        return 0, "Session expired. Login again."
    new_p = max(0, int(page or 0) - 1)
    return new_p, load_page(new_p)



# ---------------------------------------------------------------------------
# 10. UI Layout (Gradio Blocks layout)
# ---------------------------------------------------------------------------

with gr.Blocks(
    title="Admin Dashboard",
    css="""
body, .gradio-container {background:black;color:white;}
button:hover {background:#444 !important;}
#lightbox {
    display:none; position:fixed; top:0; left:0; width:100%; height:100%;
    background:rgba(0,0,0,0.85); justify-content:center; align-items:center; z-index:9999;
}
#lightbox img {max-height:90%;max-width:90%;border:3px solid #fff;}
"""
) as app:

    # Lightbox modal for full-screen previews
    gr.HTML("""
    <div id="lightbox" onclick="this.style.display='none'">
        <img id="lightbox-img">
    </div>
    """)

    gr.HTML("<h2 style='text-align:center'>Admin Login</h2>")

    # Login fields
    username = gr.Textbox(label="Username")
    password = gr.Textbox(label="Password", type="password")
    login_btn = gr.Button("Login")
    login_msg = gr.Markdown(visible=False)

    # Main table + controls (hidden until login)
    table_view = gr.HTML(visible=False)
    with gr.Row():
        refresh_btn = gr.Button("Refresh", visible=False)
        prev_btn    = gr.Button("⬅ Prev Page", visible=False)
        next_btn    = gr.Button("Next Page ➡", visible=False)
        logout_btn  = gr.Button("Logout", visible=False)

    # States for session token + current page
    token_state = gr.State("")
    page_state = gr.State(0)

    # Button bindings
    login_btn.click(
        do_login,
        inputs=[username, password],
        outputs=[token_state, login_msg, table_view, refresh_btn, logout_btn, prev_btn, next_btn, token_state]
    )

    refresh_btn.click(refresh_logs, [token_state, page_state], [table_view])
    prev_btn.click(prev_page, [token_state, page_state], [page_state, table_view])
    next_btn.click(next_page, [token_state, page_state], [page_state, table_view])
    logout_btn.click(
        logout,
        inputs=[token_state],
        outputs=[token_state, login_msg, table_view, refresh_btn, logout_btn, prev_btn, next_btn, page_state]
    )


# ---------------------------------------------------------------------------
# 12. Launch dashboard
# ---------------------------------------------------------------------------
app.launch()
