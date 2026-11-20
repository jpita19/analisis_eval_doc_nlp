from typing import List, Dict, Any
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from app.schemas.evaluation import (
    ModelPerformanceMetrics,
    GlobalDistribution,
    BenchmarkResponse,
    CategoryMetric,
)


class MetricsService:
    @staticmethod
    def calculate_performance_metrics(
        df_classified: pd.DataFrame,
        elapsed_seconds: float,
        total_original_rows: int,
        group_cols: List[str],
        class_col: str = "clasificacion",
        conf_col: str = "confianza"
    ) -> ModelPerformanceMetrics:
        total_rows = len(df_classified)
        excluded_rows = max(0, total_original_rows - total_rows)
        
        # Conteo de grupos únicos (Docente - Materia)
        unique_groups = len(df_classified.groupby(group_cols)) if total_rows > 0 else 0
        
        elapsed = max(elapsed_seconds, 0.0001)
        throughput = round(total_rows / elapsed, 2)
        latency_ms = round((elapsed / max(total_rows, 1)) * 1000, 2)

        counts = df_classified[class_col].value_counts().to_dict()
        pos_count = counts.get("Positiva", 0)
        neg_count = counts.get("Por mejorar", 0)
        na_count = counts.get("Ninguna Observación", 0)

        pos_pct = round((pos_count / total_rows) * 100, 2) if total_rows > 0 else 0.0
        neg_pct = round((neg_count / total_rows) * 100, 2) if total_rows > 0 else 0.0
        na_pct = round((na_count / total_rows) * 100, 2) if total_rows > 0 else 0.0

        distribucion = GlobalDistribution(
            positivas_total=pos_count,
            positivas_pct=pos_pct,
            por_mejorar_total=neg_count,
            por_mejorar_pct=neg_pct,
            sin_observaciones_total=na_count,
            sin_observaciones_pct=na_pct,
        )

        avg_conf_global = round(float(df_classified[conf_col].mean()), 4) if total_rows > 0 else 0.0
        
        def get_avg_conf_for(cat: str) -> float:
            sub = df_classified[df_classified[class_col] == cat]
            if len(sub) == 0:
                return 0.0
            return round(float(sub[conf_col].mean()), 4)

        return ModelPerformanceMetrics(
            total_registros_evaluados=total_rows,
            total_registros_filtrados_excluidos=excluded_rows,
            total_grupos_docente_materia=unique_groups,
            tiempo_procesamiento_segundos=round(elapsed, 4),
            registros_por_segundo=throughput,
            latencia_promedio_ms=latency_ms,
            confianza_promedio_global=avg_conf_global,
            confianza_promedio_positivas=get_avg_conf_for("Positiva"),
            confianza_promedio_por_mejorar=get_avg_conf_for("Por mejorar"),
            confianza_promedio_sin_observaciones=get_avg_conf_for("Ninguna Observación"),
            distribucion_global=distribucion,
        )

    @staticmethod
    def evaluate_benchmark(
        y_true: List[str],
        y_pred: List[str],
        elapsed_seconds: float
    ) -> BenchmarkResponse:
        total = len(y_true)
        if total == 0:
            raise ValueError("No se proporcionaron etiquetas para evaluación.")

        acc = float(accuracy_score(y_true, y_pred))
        labels = ["Positiva", "Por mejorar", "Ninguna Observación"]
        
        report = classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        )
        
        cat_metrics: Dict[str, CategoryMetric] = {}
        for label in labels:
            if label in report:
                cat_metrics[label] = CategoryMetric(
                    precision=round(report[label]["precision"], 4),
                    recall=round(report[label]["recall"], 4),
                    f1_score=round(report[label]["f1-score"], 4),
                    support=int(report[label]["support"]),
                )
            else:
                cat_metrics[label] = CategoryMetric(precision=0.0, recall=0.0, f1_score=0.0, support=0)

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_dict: Dict[str, Dict[str, int]] = {}
        for i, true_label in enumerate(labels):
            cm_dict[true_label] = {}
            for j, pred_label in enumerate(labels):
                cm_dict[true_label][pred_label] = int(cm[i][j])

        return BenchmarkResponse(
            total_muestras_evaluadas=total,
            exactitud_global_accuracy=round(acc, 4),
            metricas_por_categoria=cat_metrics,
            matriz_confusion=cm_dict,
            tiempo_evaluacion_segundos=round(elapsed_seconds, 4),
        )


metrics_service = MetricsService()
