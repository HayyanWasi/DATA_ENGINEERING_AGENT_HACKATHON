from contextlib import asynccontextmanager

from fastapi import FastAPI

from database import init_db
from models import UserDataset  # noqa: F401 — registers model with Base.metadata
from routers.import_data import router as import_data_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create database tables on startup."""
    await init_db()
    yield


app = FastAPI(title="Data Import API", lifespan=lifespan)
app.include_router(import_data_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
