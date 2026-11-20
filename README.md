# Proyecto: API de Evaluación Docente & Clasificación NLP

Sistema modular basado en **FastAPI**, **Docker** y modelos de **Procesamiento de Lenguaje Natural (NLP)** para procesar archivos de evaluación docente institucional y clasificar observaciones en: **Positivas**, **Por mejorar** y **Ninguna Observación**.

---

## 📋 Estructura de Entrada Requerida

El archivo cargado (`.csv` o `.xlsx`) debe contener las siguientes columnas:

1. **`Programa`**: Nombre del programa académico (ej. *Ingeniería de Sistemas*).
2. **`Docente`**: Nombre del docente evaluado.
3. **`Materia`**: Asignatura evaluada.
4. **`Observación`**: Texto de la opinión o sugerencia del estudiante.
5. **`Tipo de Evaluación`**: Modalidad o categoría de la evaluación.
6. **`Periodo`**: Periodo académico (ej. *2026-2*).

### 🔍 Regla de Filtrado Institucional
El sistema filtra y procesa **únicamente** los registros donde:
$$\text{Tipo de Evaluación} == \text{"DOCENCIA PLANTA- OCASIONAL"}$$
*(Las demás filas, como docencia hora cátedra o investigación, se excluyen automáticamente del reporte y se contabilizan en la hoja de métricas).*

---

## 📊 Estructura del Archivo de Salida Excel (`.xlsx`)

El archivo Excel generado contiene 3 hojas:

### Hoja 1: `Resumen_Evaluacion`
Consolidado agrupado por **Docente + Programa + Materia + Periodo** con las siguientes columnas exactas:

* **`Docente`**
* **`Programa`**
* **`Materia`**
* **`Periodo`**
* **`Total de registros`**
* **`Estudiantes con comentario`** *(Observaciones Positivas + Por mejorar)*
* **`Estudiantes sin comentario`** *(Ninguna Observación / Vacíos / NA)*
* **`Observaciones Positivas`**
* **`%Positivo`**
* **`Observaciones Por mejorar`**
* **`% Por mejorar`**
* **`Ninguna Observación`**
* **`% Sin observaciones`**
* **`Total porcentaje`** *(100.00%)*

### Hoja 2: `Detalle_Observaciones`
Registro individual fila por fila con:
* `Docente`, `Programa`, `Materia`, `Periodo`, `Tipo de Evaluación`
* `Observación` original del estudiante
* `Clasificación NLP` (*Positiva*, *Por mejorar*, *Ninguna Observación*)
* `Confianza` (porcentaje de certidumbre del modelo)

### Hoja 3: `Metricas_Procesamiento`
* Total de registros en el archivo original vs. evaluados vs. excluidos por filtro.
* Total de grupos únicos (Docente - Materia).
* Tiempos de cómputo, throughput (observaciones/segundo) y latencia.
* Confianza promedio global y por categoría.
* Distribución global de categorías.

---

## 🚀 Despliegue y Ejecución

### Opción A: Con Docker (Recomendado)
```bash
docker compose up --build
```

### Opción B: Ejecución Local
```bash
# Iniciar API con Uvicorn
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Correr suite de pruebas (11 tests unitarios y de integración)
python -m pytest
```

---

## 📡 Endpoints de la API

* **Documentación interactiva Swagger**: [http://localhost:8000/docs](http://localhost:8000/docs)
* **`POST /api/v1/evaluation/process`**: Sube el archivo `.csv` o `.xlsx` y **descarga directamente el reporte consolidado en `.xlsx`**.
* **`POST /api/v1/evaluation/analyze-json`**: Procesa y devuelve los datos consolidados y métricas en formato JSON.
* **`POST /api/v1/evaluation/benchmark`**: Evalúa la exactitud (Accuracy, F1-Score) contra etiquetas de prueba.
* **`GET /api/v1/health`**: Estado de salud de la API y del modelo NLP.
