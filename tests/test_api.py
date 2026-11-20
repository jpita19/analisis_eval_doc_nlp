import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "nlp_model" in data


def test_process_excel_endpoint():
    with open("data/samples/ejemplo_evaluacion.csv", "rb") as f:
        response = client.post(
            "/api/v1/evaluation/process",
            files={"file": ("ejemplo_evaluacion.csv", f, "text/csv")}
        )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    assert "attachment; filename=Reporte_Evaluacion_Docente_" in response.headers["content-disposition"]
    assert len(response.content) > 1000


def test_analyze_json_endpoint():
    with open("data/samples/ejemplo_evaluacion.xlsx", "rb") as f:
        response = client.post(
            "/api/v1/evaluation/analyze-json",
            files={"file": ("ejemplo_evaluacion.xlsx", f, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
        )
    assert response.status_code == 200
    data = response.json()
    assert "metricas" in data
    assert "resumen" in data
    assert data["metricas"]["total_registros_evaluados"] == 17
    assert data["metricas"]["total_registros_filtrados_excluidos"] == 3
    assert len(data["resumen"]) > 0

    first_item = data["resumen"][0]
    assert "docente" in first_item
    assert "programa" in first_item
    assert "materia" in first_item
    assert "periodo" in first_item
    assert "estudiantes_con_comentario" in first_item
    assert "estudiantes_sin_comentario" in first_item
    assert "total_porcentaje" in first_item


def test_benchmark_endpoint():
    with open("data/samples/ejemplo_evaluacion.csv", "rb") as f:
        response = client.post(
            "/api/v1/evaluation/benchmark",
            files={"file": ("ejemplo_evaluacion.csv", f, "text/csv")},
            data={"label_column": "categoria_real"}
        )
    assert response.status_code == 200
    data = response.json()
    assert "exactitud_global_accuracy" in data
    assert "metricas_por_categoria" in data
    assert "matriz_confusion" in data
    assert data["exactitud_global_accuracy"] >= 0.80
