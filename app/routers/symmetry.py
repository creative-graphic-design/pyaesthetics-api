from typing import Annotated

import cv2
import numpy as np
from fastapi import APIRouter, File, Query, UploadFile
from pyaesthetics.symmetry import getSymmetry

router = APIRouter(prefix="/symmetry", tags=["Symmetry"])


@router.post("/", response_description="degree of vertical symmetry")
async def symmetry(
    image_file: UploadFile = File(..., description="image to analyze, in RGB"),
    min_std: Annotated[
        int, Query(description="Std threshold for subsequent splitting")
    ] = 5,
    min_size: Annotated[
        int, Query(description="Size threshold for subsequent splitting, in pixel")
    ] = 20,
) -> float:
    """This function returns the degree of symmetry (0-100) between the left and right side of an image"""
    data = await image_file.read()
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    return getSymmetry(image, minStd=min_std, minSize=min_size, plot=False)
