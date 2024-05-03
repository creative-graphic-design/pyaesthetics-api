from tempfile import NamedTemporaryFile
from typing import Annotated, List, Literal, Tuple

import cv2
import numpy as np
from fastapi import APIRouter, File, Query, UploadFile
from pyaesthetics.analysis import analyzeImage, textDetection
from pydantic import BaseModel

AnalysisMethodType = Literal["complete", "fast"]

router = APIRouter(prefix="/analysis", tags=["Analysis"])


class Brightness(BaseModel):
    bt601: float
    bt709: float


class Colorfulness(BaseModel):
    hsv: float
    rgb: float


class W3CColor(BaseModel):
    color_name: str
    percentage_of_pixels: float


class AnalysisOutput(BaseModel):
    colorfulness: Colorfulness
    colors: List[W3CColor]
    faces: List[Tuple[int, int, int, int]]
    num_of_faces: int
    num_of_images: int
    symmetry_qtd: float
    text: int
    text_image_ratio: float
    vc_quad_tree: int
    vc_weight: int
    brightness: Brightness
    image_area: int
    text_area: int


@router.post(
    "/text",
    response_description="Number of character in the text",
)
async def text_detection(
    image_file: UploadFile = File(..., description="image to analyze, in RGB"),
) -> int:
    """This entrypoint uses `pytesseract` to get information about the presence of text in an image."""
    data = await image_file.read()
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return textDetection(image)


@router.post(
    "/image",
    response_description="Number of character in the text",
)
async def analyze_image(
    image_file: UploadFile = File(..., description="image to analyze, in RGB"),
    method: Annotated[
        AnalysisMethodType,
        Query(
            description="sets to analysis to use. Valid methods are `fast`, `complete`. Default is `fast`."
        ),
    ] = "complete",
    is_resize: Annotated[
        bool,
        Query(
            description="indicates wether to resize the image (reduce computational workload, increase requested time)"
        ),
    ] = True,
    new_size_w: Annotated[
        int,
        Query(
            description="if the image has to be resized, this indicates the new width of the image"
        ),
    ] = 600,
    new_size_h: Annotated[
        int,
        Query(
            description="if the image has to be resized, this indicates the new height of the image"
        ),
    ] = 400,
    min_std: Annotated[
        int,
        Query(
            description="minimum standard deviation for the Quadratic Tree Decomposition"
        ),
    ] = 10,
    min_size: Annotated[
        int, Query(description="minimum size for the Quadratic Tree Decomposition")
    ] = 20,
) -> AnalysisOutput:
    """This endpoint acts as entrypoint for the automatic analysis of an image aesthetic features."""
    with NamedTemporaryFile() as temp_file:
        content = await image_file.read()
        temp_file.write(content)

        output = analyzeImage(
            temp_file.name,
            method=method,
            resize=is_resize,
            newSize=(new_size_w, new_size_h),
            minStd=min_std,
            minSize=min_size,
        )

    return AnalysisOutput(
        colorfulness=Colorfulness(
            hsv=output["Colorfulness_HSV"], rgb=output["Colorfulness_RGB"]
        ),
        colors=[
            W3CColor(color_name=c[0], percentage_of_pixels=c[1])
            for c in output["Colors"]
        ],
        # faces=output["Faces"].tolist(),
        faces=output["Faces"],
        num_of_faces=output["Number_of_Faces"],
        num_of_images=output["Number_of_Images"],
        symmetry_qtd=output["Symmetry_QTD"],
        text=output["Text"],
        text_image_ratio=output["TextImageRatio"],
        vc_quad_tree=output["VC_quadTree"],
        vc_weight=output["VC_weight"],
        brightness=Brightness(
            bt601=output["brightness_BT601"], bt709=output["brightness_BT709"]
        ),
        image_area=output["imageArea"],
        text_area=output["textArea"],
    )
