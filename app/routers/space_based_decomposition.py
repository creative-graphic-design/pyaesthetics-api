from typing import Tuple

import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile
from pyaesthetics.spaceBasedDecomposition import getAreas, textImageRatio

router = APIRouter(
    prefix="/space-based-decomposition", tags=["Space-based decomposition"]
)


async def _get_areas(
    image_file: UploadFile = File(..., description="image to analyze, in RGB"),
    min_area: int = 100,
    is_resize: bool = True,
    new_size: Tuple[int, int] = (600, 400),
    is_plot: bool = False,
    is_coordinates: bool = False,
    is_areatype: bool = True,
):
    data = await image_file.read()
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    areas = getAreas(
        image,
        minArea=min_area,
        resize=is_resize,
        newSize=new_size,
        plot=is_plot,
        coordinates=is_coordinates,
        areatype=is_areatype,
    )
    return {
        k_area: {k: v if k != "area" else v.item() for k, v in v_area.items()}
        for k_area, v_area in areas.items()
    }


@router.post("/areas")
async def get_areas(
    image_file: UploadFile,
    min_area: int = 100,
    is_resize: bool = True,
    new_size_w: int = 600,
    new_size_h: int = 400,
    is_plot: bool = False,
    is_coordinates: bool = False,
    is_areatype: bool = True,
):

    areas = await _get_areas(
        image_file=image_file,
        min_area=min_area,
        is_resize=is_resize,
        new_size=(new_size_w, new_size_h),
        is_plot=is_plot,
        is_coordinates=is_coordinates,
        is_areatype=is_areatype,
    )
    return areas


@router.post("/text-image-ratio")
async def text_image_ratio(
    image_file: UploadFile,
    min_area: int = 100,
    is_resize: bool = True,
    new_size_w: int = 600,
    new_size_h: int = 400,
    is_plot: bool = False,
    is_coordinates: bool = False,
    is_areatype: bool = True,
):
    areas = await _get_areas(
        image_file=image_file,
        min_area=min_area,
        is_resize=is_resize,
        new_size=(new_size_w, new_size_h),
        is_plot=is_plot,
        is_coordinates=is_coordinates,
        is_areatype=is_areatype,
    )
    return textImageRatio(areas)
