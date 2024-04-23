from typing import Annotated

import cv2
import numpy as np
from fastapi import APIRouter, File, Query, UploadFile
from pyaesthetics.selfsimilarity import selfsimilarity

router = APIRouter(prefix="/self-similarity", tags=["Self similarity"])


@router.post("/", response_description="degree of self similarity")
async def self_similarity(
    image_file: UploadFile = File(..., description="image to analyze, in RGB"),
    max_level: Annotated[
        int, Query(description="Maximum number of level to analyze")
    ] = 4,
):
    data = await image_file.read()
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return selfsimilarity(image, maxlevel=max_level)
