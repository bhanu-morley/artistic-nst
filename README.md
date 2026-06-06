# Artistic Neural Style Transfer for Images using Deep Learning

> Transform ordinary photographs into stunning artistic masterpieces using Deep Learning, Neural Style Transfer, TensorFlow, Gradio, and MongoDB.

![Python](https://img.shields.io/badge/Python-3.12.9-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange)
![MongoDB](https://img.shields.io/badge/MongoDB-GridFS-green)
![Gradio](https://img.shields.io/badge/Gradio-WebUI-purple)
![License](https://img.shields.io/badge/License-Educational-red)

---

# Project Overview

Artistic Neural Style Transfer (NST) is a Deep Learning technique that combines the **content of one image** with the **artistic style of another image** to generate visually appealing artwork.

This project implements a complete end-to-end NST system that allows users to:

* Upload a Content Image
* Upload a Style Image
* Generate AI-powered artistic outputs
* Download generated artwork
* Store all generated images in MongoDB GridFS
* Monitor generated records through an Admin Dashboard

The system leverages a pre-trained Neural Style Transfer model from TensorFlow Hub and provides a responsive web interface using Gradio.

---

# Features

### Neural Style Transfer

* Combines content and style images
* Preserves image structure
* Transfers artistic textures and patterns

### Fast Processing

* Pre-trained TensorFlow model
* Optimized image preprocessing
* Cached model loading using LRU Cache

### Interactive Web Interface

* Built with Gradio
* Drag-and-drop image upload
* Live output preview
* Download generated images

### Database Integration

* MongoDB + GridFS storage
* Stores:

  * Content Image
  * Style Image
  * Generated Output
* Thumbnail generation for faster previews

### Admin Dashboard

* Secure login system
* Image preview
* Pagination support
* Record management
* Download generated outputs

### Background Processing

* Asynchronous database storage
* Non-blocking UI experience

---

# System Architecture

```text
                ┌─────────────────┐
                │     User UI     │
                │    (Gradio)     │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Image Uploads   │
                │ Content + Style │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Backend Logic   │
                │  Python API     │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ NST Model       │
                │ TensorFlow Hub  │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Stylized Image  │
                └────────┬────────┘
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
      Download Image         MongoDB GridFS
                                      │
                                      ▼
                           Admin Dashboard
```

---

# Technology Stack

| Category                | Technologies             |
| ----------------------- | ------------------------ |
| Programming Language    | Python                   |
| Deep Learning Framework | TensorFlow               |
| Pre-trained Model       | TensorFlow Hub NST Model |
| Model Serving           | TensorFlow Hub           |
| Computer Vision         | OpenCV                   |
| Image Processing        | Pillow (PIL)             |
| Numerical Computing     | NumPy                    |
| Frontend/UI             | Gradio                   |
| Database                | MongoDB                  |
| File Storage            | GridFS                   |

---

# 📂 Project Structure

```text
ARTISTIC-NST/
│
├── app.py                     # Main Gradio application
├── api.py                     # NST model loading and inference
├── admin.py                   # Admin dashboard
├── requirements.txt           # Dependencies
├── .env                       # Environment variables
├── .gitignore
│
├── src/
│   │
│   ├── database/
│   │   └── db.py             # MongoDB & GridFS operations
│   │
│   └── trained_models/
│       ├── saved_model.pb    # TensorFlow model
│       └── variables/        # Model weights
│
└── .gradio/
```

---

# Workflow

### Step 1

User uploads:

* Content Image
* Style Image

### Step 2

Backend preprocesses:

* Resize images
* Normalize pixel values
* Convert to tensors

### Step 3

TensorFlow NST model extracts:

* Content Features
* Style Features

### Step 4

Model generates:

* Stylized Artistic Output

### Step 5

Output image displayed to user

### Step 6

Images stored in MongoDB GridFS

### Step 7

Admin dashboard provides:

* Image history
* Download options
* Monitoring tools

---

# Neural Style Transfer Pipeline

```python
Content Image
      │
      ▼
Feature Extraction
      │
      ▼
Style Image
      │
      ▼
TensorFlow NST Model
      │
      ▼
Feature Fusion
      │
      ▼
Stylized Output
```

---

# Sample Use Cases

### Digital Art Creation

Convert photographs into paintings.

### Graphic Design

Generate creative visual assets.

### Social Media Content

Create unique artistic posts.

### Photography Enhancement

Apply artistic effects automatically.

### Creative Media Applications

AI-assisted artistic workflows.

---

# Database Design

### MongoDB Collections

#### Admins

Stores admin login credentials.

```json
{
  "username": "admin",
  "password_hash": "bcrypt_hash"
}
```

#### Records

Stores generated image metadata.

```json
{
  "content_file_id": "...",
  "style_file_id": "...",
  "output_file_id": "...",
  "created_at": "...",
  "format": "PNG"
}
```

---

# Admin Dashboard Features

* Secure Authentication
* Session Management
* Thumbnail Preview
* Full Image Viewer
* Pagination
* Download Support
* Record Tracking

---

# Performance Highlights

 Fast inference using TensorFlow Hub

 Efficient image preprocessing

 Cached model loading

 Responsive Gradio interface

 Asynchronous database operations

 Scalable MongoDB storage

 Optimized thumbnail generation

---

# Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/artistic-neural-style-transfer.git
cd artistic-neural-style-transfer
```

### Create Virtual Environment

```bash
python -m venv .venv
```

### Activate Environment

Windows:

```bash
.venv\Scripts\activate
```

Linux/Mac:

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Create a `.env` file:

```env
MONGO_URI=mongodb://localhost:27017
DB_NAME=artistic_nst
```

---

# Run Application

### Launch Main Application

```bash
python app.py
```

Open:

```text
http://localhost:7860
```

---

### Launch Admin Dashboard

```bash
python admin.py
```

---

# Future Enhancements

* Feed-Forward Neural Style Transfer Training
* Multi-Style Transfer Support
* Adaptive Instance Normalization (AdaIN)
* Video Style Transfer
* Cloud Deployment
* Mobile Application Integration
* User Style Intensity Controls
* Real-Time GPU Acceleration
* High-Resolution Artistic Rendering

---

## 🌟 If you found this project useful, consider giving it a star! ⭐
