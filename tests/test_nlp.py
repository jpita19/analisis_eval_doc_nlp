import pytest
from app.services.nlp_service import NLPClassifierService


@pytest.fixture
def classifier():
    return NLPClassifierService()


def test_non_informative_detection(classifier):
    assert classifier.predict_one("")[0] == "Ninguna Observación"
    assert classifier.predict_one("   ")[0] == "Ninguna Observación"
    assert classifier.predict_one("n/a")[0] == "Ninguna Observación"
    assert classifier.predict_one("Ninguna")[0] == "Ninguna Observación"
    assert classifier.predict_one("Sin comentarios")[0] == "Ninguna Observación"
    assert classifier.predict_one("...")[0] == "Ninguna Observación"


def test_positive_classification(classifier):
    cat, conf = classifier.predict_one("Excelente profesor explica muy bien y es muy atento con los alumnos")
    assert cat == "Positiva"
    assert conf > 0.5


def test_negative_classification(classifier):
    cat, conf = classifier.predict_one("Pésima atención, el docente es impuntual y nunca explica nada")
    assert cat == "Por mejorar"
    assert conf > 0.5


def test_batch_classification(classifier):
    texts = [
        "Gran profesor muy dedicado y claro",
        "Horrible clase, aburrido y desorganizado",
        "Ninguna observación",
    ]
    results = classifier.predict_batch(texts)
    assert len(results) == 3
    assert results[0][0] == "Positiva"
    assert results[1][0] == "Por mejorar"
    assert results[2][0] == "Ninguna Observación"
