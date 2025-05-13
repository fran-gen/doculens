import base64
from enum import Enum

import aiofiles


def encode_image_to_base64(image_path: str):
    """Encode the image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


async def encode_image_to_base64_async(image_path: str):
    """Encode the image to base64."""
    async with aiofiles.open(image_path, "rb") as image_file:
        image_data = await image_file.read()
        return base64.b64encode(image_data).decode("utf-8")


class GPT4VDetail(Enum):
    """An enum for the different levels of detail that the GPT-4 Vision model can provide."""

    LOW = "low"
    HIGH = "high"
    AUTO = "auto"
