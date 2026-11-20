from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.core.config import settings
from app.api.v1.router import api_router
from app.services.nlp_service import nlp_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("eval_docente")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Iniciando aplicación de Evaluación Docente...")
    logger.info(f"Modelo NLP activo: {nlp_service.model_name} (Loaded: {nlp_service.is_loaded})")
    yield
    logger.info("Deteniendo aplicación de Evaluación Docente...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="API para clasificación NLP de opiniones de estudiantes/clientes con agregación por ID y exportación a Excel (.xlsx)",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rutas
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/", tags=["General"])
async def root():
    return {
        "mensaje": "Bienvenido a la API de Evaluación y Clasificación NLP",
        "documentacion": "/docs",
        "version": settings.VERSION,
        "endpoints_principales": {
            "procesar_excel": f"{settings.API_V1_STR}/evaluation/process",
            "metricas_json": f"{settings.API_V1_STR}/evaluation/analyze-json",
            "benchmark": f"{settings.API_V1_STR}/evaluation/benchmark",
            "salud": f"{settings.API_V1_STR}/health"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
