import io
import re
import time
from typing import Tuple, List, Optional, Dict, Any
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

from app.services.nlp_service import nlp_service
from app.services.metrics_service import metrics_service
from app.schemas.evaluation import ModelPerformanceMetrics


class ReportService:
    COLUMN_SYNONYMS = {
        "programa": ["programa", "nombre_programa", "carrera", "facultad"],
        "docente": ["docente", "nombre_docente", "nombre del docente", "profesor", "profesor(a)"],
        "materia": ["materia", "asignatura", "curso", "nombre_materia"],
        "observacion": ["observacion", "observación", "opinion", "opinión", "comentario", "comentarios", "observaciones"],
        "tipo_evaluacion": ["tipo de evaluacion", "tipo de evaluación", "tipo_evaluacion", "tipo evaluacion", "tipo_de_evaluacion"],
        "periodo": ["periodo", "perido", "semestre", "anio_periodo", "ciclo"]
    }

    TARGET_EVAL_TYPE = "docencia planta-ocasional"

    @classmethod
    def read_uploaded_file(cls, file_bytes: bytes, filename: str) -> pd.DataFrame:
        lower_name = filename.lower()
        if lower_name.endswith(".csv"):
            try:
                df = pd.read_csv(io.BytesIO(file_bytes), sep=None, engine='python', encoding='utf-8')
            except Exception:
                try:
                    df = pd.read_csv(io.BytesIO(file_bytes), sep=';', encoding='latin-1')
                except Exception:
                    df = pd.read_csv(io.BytesIO(file_bytes), sep=',', encoding='latin-1')
        elif lower_name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(file_bytes))
        else:
            raise ValueError(f"Formato de archivo no soportado: '{filename}'. Debe ser .csv o .xlsx")

        if df.empty:
            raise ValueError("El archivo cargado está vacío.")

        return df

    @classmethod
    def resolve_columns(cls, df: pd.DataFrame) -> Dict[str, str]:
        cols = list(df.columns)
        cols_norm = [re.sub(r"\s+", " ", str(c).strip().lower()) for c in cols]
        resolved = {}

        for key, synonyms in cls.COLUMN_SYNONYMS.items():
            found_col = None
            for syn in synonyms:
                if syn in cols_norm:
                    found_col = cols[cols_norm.index(syn)]
                    break
            if found_col:
                resolved[key] = found_col
            else:
                raise ValueError(
                    f"No se encontró la columna requerida para '{key}'. "
                    f"Se buscaban variantes como: {synonyms}. Columnas disponibles en el archivo: {cols}"
                )

        return resolved

    @classmethod
    def filter_by_tipo_evaluacion(cls, df: pd.DataFrame, tipo_col: str) -> Tuple[pd.DataFrame, int]:
        total_original = len(df)
        
        def match_tipo(val: Any) -> bool:
            if pd.isna(val) or val is None:
                return False
            clean_val = re.sub(r"\s+", " ", str(val).strip().lower())
            clean_val = re.sub(r"\s*-\s*", "-", clean_val)
            return clean_val == cls.TARGET_EVAL_TYPE

        df_filtered = df[df[tipo_col].apply(match_tipo)].copy()
        
        if df_filtered.empty:
            raise ValueError(
                f"Ningún registro coincide con el filtro requerido: 'Tipo de Evaluación' == 'DOCENCIA PLANTA- OCASIONAL'. "
                f"Verifica los valores en la columna '{tipo_col}'."
            )

        return df_filtered, total_original

    @classmethod
    def process_dataframe(
        cls, df: pd.DataFrame, col_map: Dict[str, str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, ModelPerformanceMetrics]:
        start_time = time.time()
        
        # 1. Aplicar filtro institucional
        df_filtered, total_original = cls.filter_by_tipo_evaluacion(df, col_map["tipo_evaluacion"])

        # 2. Limpieza básica
        doc_col = col_map["docente"]
        prog_col = col_map["programa"]
        mat_col = col_map["materia"]
        per_col = col_map["periodo"]
        obs_col = col_map["observacion"]

        df_filtered[doc_col] = df_filtered[doc_col].fillna("SIN DOCENTE").astype(str).str.strip()
        df_filtered[prog_col] = df_filtered[prog_col].fillna("SIN PROGRAMA").astype(str).str.strip()
        df_filtered[mat_col] = df_filtered[mat_col].fillna("SIN MATERIA").astype(str).str.strip()
        df_filtered[per_col] = df_filtered[per_col].fillna("N/D").astype(str).str.strip()
        df_filtered[obs_col] = df_filtered[obs_col].fillna("").astype(str)

        # 3. Inferencia NLP por lotes
        texts = df_filtered[obs_col].tolist()
        predictions = nlp_service.predict_batch(texts)

        df_filtered["clasificacion"] = [p[0] for p in predictions]
        df_filtered["confianza"] = [p[1] for p in predictions]

        elapsed_time = time.time() - start_time

        # 4. Agrupación por Docente, Programa, Materia, Periodo
        group_keys = [doc_col, prog_col, mat_col, per_col]

        summary_rows = []
        for keys, group in df_filtered.groupby(group_keys):
            doc_val, prog_val, mat_val, per_val = keys
            total_reg = len(group)
            
            counts = group["clasificacion"].value_counts().to_dict()
            pos = counts.get("Positiva", 0)
            neg = counts.get("Por mejorar", 0)
            sin_obs = counts.get("Ninguna Observación", 0)

            con_comentario = pos + neg
            sin_comentario = sin_obs

            pct_pos = round((pos / total_reg) * 100, 2)
            pct_neg = round((neg / total_reg) * 100, 2)
            pct_sin = round((sin_obs / total_reg) * 100, 2)
            total_pct = round(pct_pos + pct_neg + pct_sin, 2)

            summary_rows.append({
                "Docente": doc_val,
                "Programa": prog_val,
                "Materia": mat_val,
                "Periodo": per_val,
                "Total de registros": total_reg,
                "Estudiantes con comentario": con_comentario,
                "Estudiantes sin comentario": sin_comentario,
                "Observaciones Positivas": pos,
                "%Positivo": pct_pos,
                "Observaciones Por mejorar": neg,
                "% Por mejorar": pct_neg,
                "Ninguna Observación": sin_obs,
                "% Sin observaciones": pct_sin,
                "Total porcentaje": total_pct,
            })

        df_summary = pd.DataFrame(summary_rows)
        df_summary = df_summary.sort_values(by=["Programa", "Docente", "Materia"])

        # 5. Métricas de desempeño del modelo
        metrics = metrics_service.calculate_performance_metrics(
            df_classified=df_filtered,
            elapsed_seconds=elapsed_time,
            total_original_rows=total_original,
            group_cols=group_keys
        )

        return df_summary, df_filtered, metrics

    @classmethod
    def generate_excel_report(
        cls,
        df_summary: pd.DataFrame,
        df_detail: pd.DataFrame,
        metrics: ModelPerformanceMetrics,
        col_map: Dict[str, str]
    ) -> io.BytesIO:
        output = io.BytesIO()
        wb = openpyxl.Workbook()
        wb.remove(wb.active)

        # Estilos visuales
        header_fill = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
        header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
        title_font = Font(name="Calibri", size=14, bold=True, color="1F4E79")
        subtitle_font = Font(name="Calibri", size=10, italic=True, color="595959")
        bold_font = Font(name="Calibri", size=11, bold=True)
        thin_border = Border(
            left=Side(style='thin', color='D9D9D9'),
            right=Side(style='thin', color='D9D9D9'),
            top=Side(style='thin', color='D9D9D9'),
            bottom=Side(style='thin', color='D9D9D9')
        )
        center_align = Alignment(horizontal="center", vertical="center")
        left_align = Alignment(horizontal="left", vertical="center")

        # =============================================================
        # HOJA 1: RESUMEN EVALUACIÓN DOCENTE
        # =============================================================
        ws1 = wb.create_sheet(title="Resumen_Evaluacion")
        ws1.append(["CONSOLIDADO DE EVALUACIÓN DOCENTE - NLP"])
        ws1.cell(row=1, column=1).font = title_font
        ws1.append(["Filtro aplicado: Tipo de Evaluación = 'DOCENCIA PLANTA- OCASIONAL'"])
        ws1.cell(row=2, column=1).font = subtitle_font
        ws1.append([])

        headers_s1 = [
            "Docente",
            "Programa",
            "Materia",
            "Periodo",
            "Total de registros",
            "Estudiantes con comentario",
            "Estudiantes sin comentario",
            "Observaciones Positivas",
            "%Positivo",
            "Observaciones Por mejorar",
            "% Por mejorar",
            "Ninguna Observación",
            "% Sin observaciones",
            "Total porcentaje"
        ]
        ws1.append(headers_s1)
        header_row_idx = 4
        for col_idx in range(1, len(headers_s1) + 1):
            c = ws1.cell(row=header_row_idx, column=col_idx)
            c.fill = header_fill
            c.font = header_font
            c.alignment = center_align

        for _, row in df_summary.iterrows():
            ws1.append([
                row["Docente"],
                row["Programa"],
                row["Materia"],
                row["Periodo"],
                row["Total de registros"],
                row["Estudiantes con comentario"],
                row["Estudiantes sin comentario"],
                row["Observaciones Positivas"],
                f"{row['%Positivo']:.2f}%",
                row["Observaciones Por mejorar"],
                f"{row['% Por mejorar']:.2f}%",
                row["Ninguna Observación"],
                f"{row['% Sin observaciones']:.2f}%",
                f"{row['Total porcentaje']:.2f}%",
            ])

        for r in range(5, ws1.max_row + 1):
            for c in range(1, len(headers_s1) + 1):
                cell = ws1.cell(row=r, column=c)
                cell.border = thin_border
                if c in [1, 2, 3]:
                    cell.alignment = left_align
                else:
                    cell.alignment = center_align

        cls._autofit_columns(ws1, max_len_cap=50)

        # =============================================================
        # HOJA 2: DETALLE CLASIFICADO
        # =============================================================
        ws2 = wb.create_sheet(title="Detalle_Observaciones")
        ws2.append(["DETALLE DE OBSERVACIONES CLASIFICADAS POR MODELO NLP"])
        ws2.cell(row=1, column=1).font = title_font
        ws2.append([])

        headers_s2 = [
            "Docente",
            "Programa",
            "Materia",
            "Periodo",
            "Tipo de Evaluación",
            "Observación",
            "Clasificación NLP",
            "Confianza"
        ]
        ws2.append(headers_s2)
        for col_idx in range(1, len(headers_s2) + 1):
            c = ws2.cell(row=3, column=col_idx)
            c.fill = header_fill
            c.font = header_font
            c.alignment = center_align

        doc_c = col_map["docente"]
        prog_c = col_map["programa"]
        mat_c = col_map["materia"]
        per_c = col_map["periodo"]
        tipo_c = col_map["tipo_evaluacion"]
        obs_c = col_map["observacion"]

        for _, row in df_detail.iterrows():
            ws2.append([
                str(row[doc_c]),
                str(row[prog_c]),
                str(row[mat_c]),
                str(row[per_c]),
                str(row[tipo_c]),
                str(row[obs_c]),
                str(row["clasificacion"]),
                f"{float(row['confianza']) * 100:.1f}%"
            ])

        for r in range(4, ws2.max_row + 1):
            for c in range(1, len(headers_s2) + 1):
                cell = ws2.cell(row=r, column=c)
                cell.border = thin_border
                if c == 6:
                    cell.alignment = left_align
                else:
                    cell.alignment = center_align

        cls._autofit_columns(ws2, max_len_cap=65)

        # =============================================================
        # HOJA 3: MÉTRICAS DEL PROCESAMIENTO
        # =============================================================
        ws3 = wb.create_sheet(title="Metricas_Procesamiento")
        ws3.append(["MÉTRICAS DEL MOTOR NLP Y RENDIMIENTO"])
        ws3.cell(row=1, column=1).font = title_font
        ws3.append([])

        metrics_items = [
            ("Métrica", "Valor"),
            ("Total de Registros en Archivo Original", metrics.total_registros_evaluados + metrics.total_registros_filtrados_excluidos),
            ("Registros Filtrados (DOCENCIA PLANTA- OCASIONAL)", metrics.total_registros_evaluados),
            ("Registros Excluidos (Otras Categorías)", metrics.total_registros_filtrados_excluidos),
            ("Grupos Evaluados (Docente - Materia)", metrics.total_grupos_docente_materia),
            ("Tiempo de Procesamiento", f"{metrics.tiempo_procesamiento_segundos:.4f} segundos"),
            ("Velocidad de Procesamiento", f"{metrics.registros_por_segundo:.2f} obs/seg"),
            ("Latencia Promedio por Observación", f"{metrics.latencia_promedio_ms:.2f} ms"),
            ("Confianza Promedio Global", f"{metrics.confianza_promedio_global * 100:.2f}%"),
            ("Confianza Promedio (Positivas)", f"{metrics.confianza_promedio_positivas * 100:.2f}%"),
            ("Confianza Promedio (Por mejorar)", f"{metrics.confianza_promedio_por_mejorar * 100:.2f}%"),
            ("Confianza Promedio (Sin observaciones)", f"{metrics.confianza_promedio_sin_observaciones * 100:.2f}%"),
            ("", ""),
            ("DISTRIBUCIÓN TOTAL DE CATEGORÍAS", ""),
            ("Observaciones Positivas", f"{metrics.distribucion_global.positivas_total} ({metrics.distribucion_global.positivas_pct:.2f}%)"),
            ("Observaciones Por mejorar", f"{metrics.distribucion_global.por_mejorar_total} ({metrics.distribucion_global.por_mejorar_pct:.2f}%)"),
            ("Sin observaciones / Ninguna", f"{metrics.distribucion_global.sin_observaciones_total} ({metrics.distribucion_global.sin_observaciones_pct:.2f}%)"),
        ]

        for idx, (label, val) in enumerate(metrics_items, start=3):
            ws3.append([label, val])
            if idx == 3:
                ws3.cell(row=idx, column=1).fill = header_fill
                ws3.cell(row=idx, column=1).font = header_font
                ws3.cell(row=idx, column=2).fill = header_fill
                ws3.cell(row=idx, column=2).font = header_font
            elif label == "DISTRIBUCIÓN TOTAL DE CATEGORÍAS":
                ws3.cell(row=idx, column=1).font = bold_font
            else:
                ws3.cell(row=idx, column=1).border = thin_border
                ws3.cell(row=idx, column=2).border = thin_border

        cls._autofit_columns(ws3, max_len_cap=55)

        wb.save(output)
        output.seek(0)
        return output

    @staticmethod
    def _autofit_columns(ws, max_len_cap: int = 50):
        for col in ws.columns:
            max_len = 0
            col_letter = get_column_letter(col[0].column)
            for cell in col:
                val = str(cell.value or "")
                if len(val) > max_len and cell.row > 2:
                    max_len = len(val)
            adjusted_width = min(max(max_len + 4, 12), max_len_cap)
            ws.column_dimensions[col_letter].width = adjusted_width


report_service = ReportService()
