import time
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from fastapi.responses import StreamingResponse

from app.services.report_service import report_service
from app.services.metrics_service import metrics_service
from app.services.nlp_service import nlp_service
from app.schemas.evaluation import (
    EvaluationProcessResponse,
    BenchmarkResponse,
    DocenteEvaluationSummary,
    ObservationDetail,
)

router = APIRouter()


@router.post(
    "/process",
    summary="Procesar archivo y descargar reporte institucional en Excel (.xlsx)",
    description=(
        "Sube un archivo .csv o .xlsx con columnas: Programa, Docente, Materia, Observación, Tipo de Evaluación, Periodo. "
        "Filtra automáticamente por 'DOCENCIA PLANTA- OCASIONAL' y devuelve un Excel consolidado con resumen y métricas."
    )
)
async def process_and_download_excel(
    file: UploadFile = File(..., description="Archivo .csv o .xlsx de evaluación docente")
):
    try:
        contents = await file.read()
        df = report_service.read_uploaded_file(contents, file.filename)
        col_map = report_service.resolve_columns(df)

        df_summary, df_detail, metrics = report_service.process_dataframe(
            df=df, col_map=col_map
        )

        excel_stream = report_service.generate_excel_report(
            df_summary=df_summary,
            df_detail=df_detail,
            metrics=metrics,
            col_map=col_map,
        )

        filename_out = f"Reporte_Evaluacion_Docente_{int(time.time())}.xlsx"
        return StreamingResponse(
            excel_stream,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={filename_out}"}
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error durante el procesamiento del archivo: {str(e)}"
        )


@router.post(
    "/analyze-json",
    response_model=EvaluationProcessResponse,
    summary="Procesar archivo y retornar métricas y resumen en JSON",
    description="Útil para dashboards y análisis programático de las evaluaciones docentes."
)
async def process_and_get_json(
    file: UploadFile = File(..., description="Archivo .csv o .xlsx")
):
    try:
        contents = await file.read()
        df = report_service.read_uploaded_file(contents, file.filename)
        col_map = report_service.resolve_columns(df)

        df_summary, df_detail, metrics = report_service.process_dataframe(
            df=df, col_map=col_map
        )

        resumen_list = [
            DocenteEvaluationSummary(
                docente=row["Docente"],
                programa=row["Programa"],
                materia=row["Materia"],
                periodo=row["Periodo"],
                total_registros=int(row["Total de registros"]),
                estudiantes_con_comentario=int(row["Estudiantes con comentario"]),
                estudiantes_sin_comentario=int(row["Estudiantes sin comentario"]),
                observaciones_positivas=int(row["Observaciones Positivas"]),
                pct_positivo=float(row["%Positivo"]),
                observaciones_por_mejorar=int(row["Observaciones Por mejorar"]),
                pct_por_mejorar=float(row["% Por mejorar"]),
                ninguna_observacion=int(row["Ninguna Observación"]),
                pct_sin_observaciones=float(row["% Sin observaciones"]),
                total_porcentaje=float(row["Total porcentaje"]),
            )
            for _, row in df_summary.iterrows()
        ]

        doc_c = col_map["docente"]
        prog_c = col_map["programa"]
        mat_c = col_map["materia"]
        per_c = col_map["periodo"]
        obs_c = col_map["observacion"]

        sample_detail = [
            ObservationDetail(
                docente=str(row[doc_c]),
                programa=str(row[prog_c]),
                materia=str(row[mat_c]),
                periodo=str(row[per_c]),
                observacion=str(row[obs_c]),
                clasificacion=str(row["clasificacion"]),
                confianza=float(row["confianza"]),
            )
            for _, row in df_detail.head(10).iterrows()
        ]

        return EvaluationProcessResponse(
            mensaje="Archivo procesado y filtrado exitosamente.",
            metricas=metrics,
            resumen=resumen_list,
            detalle_primeras_filas=sample_detail,
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error durante el procesamiento: {str(e)}"
        )


@router.post(
    "/benchmark",
    response_model=BenchmarkResponse,
    summary="Evaluar precisión del modelo contra columna de referencia",
    description="Compara la clasificación contra etiquetas reales (Positiva, Por mejorar, Ninguna Observación)."
)
async def evaluate_model_benchmark(
    file: UploadFile = File(..., description="Archivo .csv o .xlsx"),
    label_column: str = Form("categoria_real", description="Nombre de la columna con la etiqueta real"),
):
    try:
        contents = await file.read()
        df = report_service.read_uploaded_file(contents, file.filename)
        
        if label_column not in df.columns:
            raise ValueError(f"Columna de etiquetas '{label_column}' no encontrada en el archivo.")

        # Buscar columna de observacion
        obs_col = None
        for cand in report_service.COLUMN_SYNONYMS["observacion"]:
            for c in df.columns:
                if cand == str(c).strip().lower():
                    obs_col = c
                    break
            if obs_col:
                break
        if not obs_col:
            raise ValueError("No se encontró columna para el texto de la observación.")

        start_t = time.time()
        texts = df[obs_col].fillna("").astype(str).tolist()
        predictions = nlp_service.predict_batch(texts)
        y_pred = [p[0] for p in predictions]
        elapsed = time.time() - start_t

        y_true = df[label_column].fillna("Ninguna Observación").astype(str).tolist()
        
        y_true_normalized = []
        for val in y_true:
            v_lower = val.strip().lower()
            if "pos" in v_lower or "buen" in v_lower:
                y_true_normalized.append("Positiva")
            elif "mal" in v_lower or "mejor" in v_lower:
                y_true_normalized.append("Por mejorar")
            else:
                y_true_normalized.append("Ninguna Observación")

        benchmark_res = metrics_service.evaluate_benchmark(
            y_true=y_true_normalized,
            y_pred=y_pred,
            elapsed_seconds=elapsed,
        )
        return benchmark_res
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error durante el benchmark: {str(e)}"
        )
