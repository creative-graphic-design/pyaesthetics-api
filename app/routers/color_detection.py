from typing import List, Literal

import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile
from pyaesthetics.colorDetection import getColorsW3C
from pydantic import BaseModel

AnalysisMethodType = Literal["complete", "fast"]

router = APIRouter(prefix="/color-detection", tags=["Color detection"])


class W3CColor(BaseModel):
    color_name: str
    percentage_of_pixels: float


@router.post(
    "/",
    response_description="Percentage distribution of colors according to the W3C sixteens basic colors",
)
async def get_color_w3c(
    image_file: UploadFile = File(..., description="image to analyze, in RGB"),
) -> List[W3CColor]:
    """This endpoint is used to get a simplified color palette (W3C siteens basic colors).

    F = 255
    C0 = 192
    80 = 128
    """
    data = await image_file.read()
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    colors = getColorsW3C(image, plot=False)
    return [W3CColor(color_name=c[0], percentage_of_pixels=c[1]) for c in colors]  # type: ignore
