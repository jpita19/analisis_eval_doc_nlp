from fastapi import APIRouter
from app.api.v1.endpoints import evaluation, health

api_router = APIRouter()
api_router.include_router(evaluation.router, prefix="/evaluation", tags=["Evaluación y Clasificación NLP"])
api_router.include_router(health.router, tags=["Salud del Sistema"])
