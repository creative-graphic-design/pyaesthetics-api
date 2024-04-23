from typing import List, Tuple

import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile
from pyaesthetics.faceDetection import getFaces

router = APIRouter(prefix="/face-detection", tags=["Face detection"])


@router.post("/", response_description="Number of faces in the image")
async def get_faces(
    image_file: UploadFile = File(..., description="image to analyze, in RGB")
) -> List[Tuple[int, int, int, int]]:
    data = await image_file.read()
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    output = getFaces(image, plot=False)
    return output.tolist()  # type: ignore
