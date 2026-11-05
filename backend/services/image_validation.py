
from dataclasses import dataclass
from io import BytesIO

from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

#may need to change based on needs from the CNN
MAX_FILE_SIZE = 5 * 1024 * 1024
MAX_DIMENSION = 6000
MAX_PIXELS = 20_000_000

ALLOWED_FORMATS = {
    "image/jpeg": "JPEG",
    "image/png": "PNG",
    "image/webp": "WEBP",
}


@dataclass
class ValidatedImage:
    data: bytes
    image_format: str
    content_type: str
    width: int
    height: int


def validate_image(upload: UploadFile) -> ValidatedImage:

    # 1. Check declared file type
    if upload.content_type not in ALLOWED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail="Only JPEG, PNG, and WebP images are allowed."
        )

    # 2. Read at most 5 MB + 1 byte
    data = upload.file.read(MAX_FILE_SIZE + 1)

    if not data:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty."
        )

    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="The image must be 5 MB or smaller."
        )

    try:
        # 3. Inspect actual image format and dimensions
        with Image.open(BytesIO(data)) as img:
            image_format = img.format
            width, height = img.size

            if image_format != ALLOWED_FORMATS[upload.content_type]:
                raise HTTPException(
                    status_code=400,
                    detail="Image content does not match its file type."
                )

            if (
                width > MAX_DIMENSION
                or height > MAX_DIMENSION
                or width * height > MAX_PIXELS
            ):
                raise HTTPException(
                    status_code=413,
                    detail="Image dimensions are too large."
                )

            img.verify()

        # 4. Reopen and fully decode the image
        with Image.open(BytesIO(data)) as img:
            img.load()

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
    ) as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image file."
        ) from exc

    return ValidatedImage(
        data=data,
        image_format=image_format,
        content_type=upload.content_type,
        width=width,
        height=height,
    )
