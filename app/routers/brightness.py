import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile
from pyaesthetics.brightness import (
    relativeLuminance_BT601,
    relativeLuminance_BT709,
    sRGB2RGB,
)

router = APIRouter(prefix="/brightness", tags=["Brightness"])


@router.post(
    "/relative-luminance/bt601",
    response_description="Mean brightness based on BT.601 standard.",
)
async def get_relative_luminance_bt601(
    image_file: UploadFile = File(..., description="image to analyze, in RGB")
) -> float:
    """This endpoint evaluates the brightness of an image by mean of Y, where Y is evaluated as:

    Y = 0.7152G + 0.0722B + 0.2126R
    B = mean(Y)
    """
    data = await image_file.read()
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    image = sRGB2RGB(image)
    return relativeLuminance_BT601(image)


@router.post(
    "/relative-luminance/bt709",
    response_description="Mean brightness based on BT.709 standard.",
)
async def get_relative_luminance_bt709(
    image_file: UploadFile = File(..., description="image to analyze, in RGB")
) -> float:
    """This endpoint evaluates the brightness of an image by mean of Y, where Y is evaluated as:

    Y = 0.587G + 0.114B + 0.299R
    B = mean(Y)
    """
    data = await image_file.read()
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    image = sRGB2RGB(image)
    return relativeLuminance_BT709(image)
