from fastapi import APIRouter
from app.services.nlp_service import nlp_service
from app.core.config import settings

router = APIRouter()


@router.get("/health", summary="Estado del servicio y del modelo NLP")
async def health_check():
    return {
        "status": "healthy",
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "nlp_model": {
            "name": nlp_service.model_name,
            "pipeline_loaded": nlp_service.is_loaded,
            "engine": "HuggingFace Transformer" if nlp_service.is_loaded else "Spanish Lexicon/Heuristic Engine"
        }
    }
