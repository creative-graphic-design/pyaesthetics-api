from tempfile import NamedTemporaryFile
from typing import Annotated, List, Literal, Tuple

import cv2
import numpy as np
from fastapi import APIRouter, File, Query, UploadFile
from pyaesthetics.analysis import analyzeImage, textDetection
from pyaesthetics.selfsimilarity import selfsimilarity
from pydantic import BaseModel

AnalysisMethodType = Literal["complete", "fast"]

router = APIRouter(prefix="/analysis", tags=["Analysis"])


class Analysis(BaseModel):
    text: int


class Brightness(BaseModel):
    bt601: float
    bt709: float


class Colorfulness(BaseModel):
    hsv: float
    rgb: float


class W3CColor(BaseModel):
    color_name: str
    percentage_of_pixels: float


class FaceDetection(BaseModel):
    faces: List[Tuple[int, int, int, int]]
    num_faces: int


class QuadTreeDecomposition(BaseModel):
    vc_quad_tree: int
    vc_weight: int


class SpaceBasedDecomposition(BaseModel):
    num_images: int
    text_image_ratio: float
    text_area: int
    image_area: int


class ImageAnalysisOutput(BaseModel):
    analysis: Analysis
    brightness: Brightness
    color_detection: List[W3CColor]
    colorfulness: Colorfulness
    face_detection: FaceDetection
    quad_tree_decomposition: QuadTreeDecomposition
    space_based_decomposition: SpaceBasedDecomposition
    symmetry: float


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
            description="sets to analysis to use. Valid methods are `fast`, `complete`. Default is `complete`."
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
    max_level: Annotated[
        int, Query(description="Maximum number of level to analyze")
    ] = 4,
) -> ImageAnalysisOutput:
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

    return ImageAnalysisOutput(
        analysis=Analysis(text=output["Text"]),
        brightness=Brightness(
            bt601=output["brightness_BT601"], bt709=output["brightness_BT709"]
        ),
        color_detection=[
            W3CColor(color_name=c[0], percentage_of_pixels=c[1])
            for c in output["Colors"]
        ],
        colorfulness=Colorfulness(
            hsv=output["Colorfulness_HSV"], rgb=output["Colorfulness_RGB"]
        ),
        face_detection=FaceDetection(
            faces=output["Faces"], num_faces=output["Number_of_Faces"]
        ),
        quad_tree_decomposition=QuadTreeDecomposition(
            vc_quad_tree=output["VC_quadTree"], vc_weight=output["VC_weight"]
        ),
        space_based_decomposition=SpaceBasedDecomposition(
            num_images=output["Number_of_Images"],
            text_image_ratio=output["TextImageRatio"],
            text_area=output["textArea"],
            image_area=output["imageArea"],
        ),
        symmetry=output["Symmetry_QTD"],
    )
