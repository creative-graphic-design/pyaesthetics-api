from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from app.routers import (
    analysis,
    brightness,
    color_detection,
    colorfulness,
    face_detection,
    quad_tree_decomposition,
    self_similarity,
    space_based_decomposition,
    symmetry,
)

app = FastAPI(
    license_info={
        "name": "GPL-3.0",
        "url": "https://www.gnu.org/licenses/gpl-3.0.html",
    }
)


routers = [
    analysis.router,
    brightness.router,
    color_detection.router,
    colorfulness.router,
    face_detection.router,
    quad_tree_decomposition.router,
    space_based_decomposition.router,
    self_similarity.router,
    symmetry.router,
]
for router in routers:
    app.include_router(router)


@app.get("/", include_in_schema=False)
async def docs_redirect():
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["Health check"])
def health():
    return {"status": "OK"}
