import pandas as pd
import openpyxl
from app.services.report_service import report_service


def test_resolve_columns():
    df = pd.DataFrame(columns=[
        "Programa", "Docente", "Materia", "Observación", "Tipo de Evaluación", "Periodo"
    ])
    col_map = report_service.resolve_columns(df)
    assert col_map["docente"] == "Docente"
    assert col_map["materia"] == "Materia"
    assert col_map["tipo_evaluacion"] == "Tipo de Evaluación"


def test_filter_by_tipo_evaluacion():
    df = pd.DataFrame({
        "Docente": ["Prof A", "Prof B", "Prof C"],
        "Tipo de Evaluación": [
            "DOCENCIA PLANTA- OCASIONAL",
            "docencia planta - ocasional",
            "DOCENCIA HORA CATEDRA"
        ]
    })
    filtered, total_orig = report_service.filter_by_tipo_evaluacion(df, "Tipo de Evaluación")
    assert total_orig == 3
    assert len(filtered) == 2  # Solo los dos de planta-ocasional


def test_process_dataframe_and_excel():
    with open("data/samples/ejemplo_evaluacion.csv", "rb") as f:
        df = report_service.read_uploaded_file(f.read(), "ejemplo.csv")
    
    col_map = report_service.resolve_columns(df)
    df_summary, df_detail, metrics = report_service.process_dataframe(df, col_map)

    # El archivo original tiene 20 registros, 17 son DOCENCIA PLANTA- OCASIONAL y 3 de otras categorías
    assert len(df_detail) == 17
    assert metrics.total_registros_filtrados_excluidos == 3

    # Verificar columnas del resumen
    expected_cols = [
        "Docente", "Programa", "Materia", "Periodo",
        "Total de registros", "Estudiantes con comentario", "Estudiantes sin comentario",
        "Observaciones Positivas", "%Positivo",
        "Observaciones Por mejorar", "% Por mejorar",
        "Ninguna Observación", "% Sin observaciones",
        "Total porcentaje"
    ]
    for col in expected_cols:
        assert col in df_summary.columns

    # Generar Excel y verificar hojas
    excel_stream = report_service.generate_excel_report(df_summary, df_detail, metrics, col_map)
    wb = openpyxl.load_workbook(excel_stream)
    assert "Resumen_Evaluacion" in wb.sheetnames
    assert "Detalle_Observaciones" in wb.sheetnames
    assert "Metricas_Procesamiento" in wb.sheetnames
