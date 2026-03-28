from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from database import init_db
from models import UserDataset  # noqa: F401 — registers model with Base.metadata
from routers.import_data import router as import_data_router
from app.api import analyze_router, strategize_router, clean_router, direct_clean_router
from app.api.pipeline import router as pipeline_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create database tables on startup."""
    await init_db()
    yield


app = FastAPI(title="Data Engineering Agent API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(import_data_router)
app.include_router(analyze_router)
app.include_router(strategize_router)
app.include_router(clean_router)
app.include_router(direct_clean_router)  # no-DB direct test route
app.include_router(pipeline_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
