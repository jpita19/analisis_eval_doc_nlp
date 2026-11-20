from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any


class ObservationDetail(BaseModel):
    docente: str = Field(..., description="Nombre del docente")
    programa: str = Field(..., description="Programa académico")
    materia: str = Field(..., description="Nombre de la materia")
    periodo: str = Field(..., description="Periodo académico")
    observacion: str = Field(..., description="Texto de la observación")
    clasificacion: str = Field(..., description="Categoría asignada: Positiva, Por mejorar, Ninguna Observación")
    confianza: float = Field(..., description="Score de confianza del modelo (0.0 a 1.0)")


class DocenteEvaluationSummary(BaseModel):
    docente: str
    programa: str
    materia: str
    periodo: str
    total_registros: int
    estudiantes_con_comentario: int
    estudiantes_sin_comentario: int
    observaciones_positivas: int
    pct_positivo: float
    observaciones_por_mejorar: int
    pct_por_mejorar: float
    ninguna_observacion: int
    pct_sin_observaciones: float
    total_porcentaje: float = 100.0


class GlobalDistribution(BaseModel):
    positivas_total: int
    positivas_pct: float
    por_mejorar_total: int
    por_mejorar_pct: float
    sin_observaciones_total: int
    sin_observaciones_pct: float


class ModelPerformanceMetrics(BaseModel):
    total_registros_evaluados: int
    total_registros_filtrados_excluidos: int
    total_grupos_docente_materia: int
    tiempo_procesamiento_segundos: float
    registros_por_segundo: float
    latencia_promedio_ms: float
    confianza_promedio_global: float
    confianza_promedio_positivas: float
    confianza_promedio_por_mejorar: float
    confianza_promedio_sin_observaciones: float
    distribucion_global: GlobalDistribution


class EvaluationProcessResponse(BaseModel):
    mensaje: str
    filtro_aplicado: str = "Tipo de Evaluación == 'DOCENCIA PLANTA- OCASIONAL'"
    metricas: ModelPerformanceMetrics
    resumen: List[DocenteEvaluationSummary]
    detalle_primeras_filas: Optional[List[ObservationDetail]] = None


class CategoryMetric(BaseModel):
    precision: float
    recall: float
    f1_score: float
    support: int


class BenchmarkResponse(BaseModel):
    total_muestras_evaluadas: int
    exactitud_global_accuracy: float
    metricas_por_categoria: Dict[str, CategoryMetric]
    matriz_confusion: Dict[str, Dict[str, int]]
    tiempo_evaluacion_segundos: float
