from typing import Annotated

import cv2
import numpy as np
from fastapi import APIRouter, File, Query, Response, UploadFile
from pyaesthetics.quadTreeDecomposition import quadTree

router = APIRouter(prefix="/quad-tree-decomposition", tags=["Quad-tree decomposition"])


@router.post(
    "/", response_class=Response, response_description="Quad-tree decomposition"
)
async def quad_tree_decomposition(
    image_file: UploadFile = File(..., description="image to analyze, in RGB"),
    min_std: Annotated[
        int, Query(description="Std threshold for subsequent splitting")
    ] = 15,
    min_size: Annotated[
        int, Query(description="Size threshold for subsequent splitting, in pixel")
    ] = 40,
) -> Response:
    data = await image_file.read()
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    quad_tree = quadTree(gray_image, min_std, min_size)

    for block in quad_tree.blocks:
        cv2.rectangle(
            image,
            (block[0], block[1]),
            (block[0] + block[3], block[1] + block[2]),
            (0, 0, 255),
        )

    _, byte_image = cv2.imencode(".png", image)
    return Response(content=byte_image.tobytes(), media_type="image/png")
