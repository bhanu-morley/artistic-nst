# Database handler to store records in the MongoDB database
import io
import uuid
import bcrypt
from datetime import datetime, timezone
from typing import List, Dict
from zoneinfo import ZoneInfo

from pymongo import MongoClient, ASCENDING
from gridfs import GridFS
from PIL import Image


class NSTDatabase:
    def __init__(self, mongo_uri: str, db_name: str = "artistic_nst"):
        # Connect to MongoDB
        self.client = MongoClient(mongo_uri)

        # Select database
        self.db = self.client[db_name]

        # GridFS instance for storing images
        self.fs = GridFS(self.db)

        # Collection for storing style transfer logs/records
        self.records = self.db["records"]

        # Collection for admin login credentials
        self.admins = self.db["admins"]

        # Create indexes for faster queries
        self.records.create_index([("created_at", ASCENDING)])
        self.records.create_index([("output_file_id", ASCENDING)])
        self.admins.create_index([("username", ASCENDING)], unique=True)

    # -----------------------------------------------------------------------
    # 1. Admin Authentication
    # -----------------------------------------------------------------------
    def verify_admin(self, username: str, password: str) -> bool:
        """
        Verify admin login using stored BCrypt password hash.
        """
        # Search admin document by username
        doc = self.admins.find_one({"username": username})
        if not doc:
            return False

        # Compare hashed password
        return bcrypt.checkpw(password.encode("utf-8"), doc["password_hash"])

    # -----------------------------------------------------------------------
    # 2. Helper: Convert PIL Image → Bytes
    # -----------------------------------------------------------------------
    @staticmethod
    def _pil_to_bytes(pil_img: Image.Image, fmt: str = "PNG", quality: int = 95) -> bytes:
        """
        Convert a PIL image to raw bytes buffer for storing in GridFS.
        Supports PNG & JPEG with compression.
        """
        bio = io.BytesIO()

        # Save as JPG/JPEG with compression
        if fmt.upper() in ("JPG", "JPEG"):
            pil_img.save(bio, format=fmt, quality=quality, optimize=True)

        # Save as PNG (lossless)
        else:
            pil_img.save(bio, format="PNG", optimize=True)

        bio.seek(0)
        return bio.getvalue()

    # -----------------------------------------------------------------------
    # 3. Helper: Validate File Size
    # -----------------------------------------------------------------------
    @staticmethod
    def _bytes_size_ok(b: bytes, max_mb: int = 50) -> bool:
        """
        Ensure the image file size does not exceed 50MB.
        """
        return len(b) <= max_mb * 1024 * 1024

    # -----------------------------------------------------------------------
    # 4. Helper: Create Thumbnail
    # -----------------------------------------------------------------------
    def _make_thumb(self, pil_img, size=(256,256)):
        """
        Create a smaller thumbnail version (default 256x256)
        to speed up admin dashboard preview.
        """
        img = pil_img.copy()
        img.thumbnail(size)
        return self._pil_to_bytes(img, fmt="PNG")

    # -----------------------------------------------------------------------
    # 5. Save Content + Style + Output images
    # -----------------------------------------------------------------------
    def save_triplet(self, content_img: Image.Image, style_img: Image.Image, output_img: Image.Image, fmt="PNG") -> str:
        """
        Stores original & thumbnail versions of all 3 images in GridFS.
        Creates a record document linking them.
        """

        # Convert PIL images to byte streams
        c_bytes = self._pil_to_bytes(content_img, fmt=fmt)
        s_bytes = self._pil_to_bytes(style_img, fmt=fmt)
        o_bytes = self._pil_to_bytes(output_img, fmt=fmt)

        # Validate size limit
        for name, data in (("content", c_bytes), ("style", s_bytes), ("output", o_bytes)):
            if not self._bytes_size_ok(data):
                raise ValueError(f"{name} image exceeds 50MB limit")

        # Create thumbnails for faster UI previews
        c_thumb = self._make_thumb(content_img)
        s_thumb = self._make_thumb(style_img)
        o_thumb = self._make_thumb(output_img)

        # Save original images in GridFS
        c_id = self.fs.put(c_bytes, filename=f"content_{uuid.uuid4().hex}.png")
        s_id = self.fs.put(s_bytes, filename=f"style_{uuid.uuid4().hex}.png")
        o_id = self.fs.put(o_bytes, filename=f"output_{uuid.uuid4().hex}.png")

        # Save thumbnails in GridFS
        c_t_id = self.fs.put(c_thumb, filename=f"thumb_content_{uuid.uuid4().hex}.png")
        s_t_id = self.fs.put(s_thumb, filename=f"thumb_style_{uuid.uuid4().hex}.png")
        o_t_id = self.fs.put(o_thumb, filename=f"thumb_output_{uuid.uuid4().hex}.png")

        # Save record in DB with India Standard Time timestamp
        ts_ist = datetime.now(ZoneInfo("Asia/Kolkata"))

        rec = {
            "content_file_id": c_id,
            "style_file_id": s_id,
            "output_file_id": o_id,
            "content_thumb_id": c_t_id,
            "style_thumb_id": s_t_id,
            "output_thumb_id": o_t_id,
            "created_at": ts_ist,
            "format": fmt,
        }

        ins = self.records.insert_one(rec)
        return str(ins.inserted_id)

    # -----------------------------------------------------------------------
    # 6. Fetch list for admin UI
    # -----------------------------------------------------------------------
    def list_records(self, skip: int = 0, limit: int = 20) -> List[Dict]:
        """
        Returns paginated list of generated images.
        Converts timestamps to IST + formatted string.
        Falls back to original files if thumbnail missing.
        """

        cur = self.records.find({}, sort=[("created_at", -1)]).skip(skip).limit(limit)

        out = []
        for d in cur:
            dt = d["created_at"]

            # Handle old DB entries without timezone info
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)

            dt_ist = dt.astimezone(ZoneInfo("Asia/Kolkata"))
            created_str = dt_ist.strftime("%d-%b-%Y %I:%M:%S %p")

            out.append({
                "_id": str(d["_id"]),
                "created_at": created_str,
                "format": d.get("format", "PNG"),

                # Prefer thumbnails, fallback to original
                "content_thumb_id": str(d.get("content_thumb_id", d["content_file_id"])),
                "style_thumb_id": str(d.get("style_thumb_id", d["style_file_id"])),
                "output_thumb_id": str(d.get("output_thumb_id", d["output_file_id"])),

                "content_file_id": str(d["content_file_id"]),
                "style_file_id": str(d["style_file_id"]),
                "output_file_id": str(d["output_file_id"]),
            })
        return out

    # -----------------------------------------------------------------------
    # 7. Load Image Bytes by GridFS File ID
    # -----------------------------------------------------------------------
    def get_file_bytes(self, file_id_str: str) -> bytes:
        """
        Fetch stored image bytes from GridFS by ObjectID.
        Used when viewing images from admin panel.
        """
        from bson import ObjectId
        return self.fs.get(ObjectId(file_id_str)).read()
