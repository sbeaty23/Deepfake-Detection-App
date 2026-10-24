
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from io import BytesIO

app = FastAPI(title="Deepfake Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173",
                   "http://127.0.0.1:5173"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

MAX_FILE_SIZE = 5 * 1024 * 1024

ALLOWED_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}


@app.post("/api/analyze")
async def analyze_image(image: UploadFile = File(...)):
    if image.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Only JPEG, PNG, and WebP images are allowed."
        )

    image_bytes = await image.read(MAX_FILE_SIZE + 1)

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty."
        )

    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="The image must be 5 MB or smaller."
        )

    try:
        with Image.open(BytesIO(image_bytes)) as img:
            if img.format not in {"JPEG", "PNG", "WEBP"}:
                raise HTTPException(
                    status_code=400,
                    detail="Unsupported image format."
                )

            img.verify()

    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image file."
        )

    # TODO: the AI-image detection model should be here.

    return {
        "success": True,
        "message": "Image uploaded and validated successfully.",
        "filename": image.filename,
        "detection_status": "not_configured",
    }
