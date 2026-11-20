import re
import logging
from typing import List, Tuple, Dict, Any, Optional
from app.core.config import settings

logger = logging.getLogger(__name__)

# Palabras y frases comunes que claramente representan "Ninguna Observación"
NON_INFORMATIVE_PATTERNS = {
    "", "n/a", "na", "no aplica", "no", "ninguna", "ninguno", "ningun comentario",
    "ninguna observacion", "ninguna observación", "sin comentarios", "sin comentario",
    "sin observaciones", "sin observación", "nada", "todo bien", "ok", "bien",
    "ningun problema", "ningún problema", "no tengo", "no hay comentarios",
    "no aplica.", "ninguna.", "nada.", "s/c", "ninguno.", "ningun aporte",
    ".", "-", "--", "...", "ninguna.", "ninguna sugerencia", "sin sugerencias",
    "ninguno todo bien", "no tengo comentarios"
}

POSITIVE_WORDS = {
    "excelente", "bueno", "buena", "buenas", "buenos", "genial", "gran", "destacado",
    "destacada", "maravilloso", "maravillosa", "cumple", "atento", "atenta",
    "claro", "clara", "explica bien", "muy bien", "dedicado", "dedicada",
    "puntual", "responsable", "comprometido", "comprometida", "amable", "respetuoso",
    "respetuosa", "recomiendo", "recomendado", "recomendada", "dinamico", "dinamica",
    "excelentes", "claridad", "aprende", "aprendi", "aprendí", "agradable", "didactico",
    "didactica", "didáctica", "didáctico", "facilitador", "facilitadora", "motivador",
    "motivadora", "correcto", "correcta", "satisfecho", "satisfecha", "positivo",
    "positiva", "excepcional", "domina", "buen docente", "gran docente", "excelente docente"
}

NEGATIVE_WORDS = {
    "malo", "mala", "malos", "malas", "pesimo", "pesima", "pésimo", "pésima", "terrible",
    "horrible", "impuntual", "falta", "grosero", "grosera", "aburrido", "aburrida",
    "injusto", "injusta", "confuso", "confusa", "desorganizado", "desorganizada",
    "no explica", "no responde", "no cumple", "mal", "nunca asiste", "tarde",
    "ineficiente", "deficiente", "incompetente", "irresponsable", "desinteresado",
    "desinteresada", "perdi el tiempo", "desagradable", "flojo", "floja", "poco claro",
    "poca claridad", "no prepara", "muy malo", "muy mala", "no asiste", "no califica",
    "queja", "por mejorar", "debe mejorar", "mejorar", "falta preparacion", "falta metodología"
}


class NLPClassifierService:
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.DEFAULT_MODEL_NAME
        self.pipeline = None
        self.is_loaded = False
        self._initialize_model()

    def _initialize_model(self):
        try:
            from transformers import pipeline
            import torch
            device = 0 if torch.cuda.is_available() else -1
            logger.info(f"Cargando modelo NLP '{self.model_name}' en dispositivo: {'GPU' if device == 0 else 'CPU'}...")
            self.pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                top_k=None,
                device=device,
                batch_size=settings.BATCH_SIZE
            )
            self.is_loaded = True
            logger.info("Modelo NLP cargado exitosamente.")
        except Exception as e:
            logger.warning(
                f"No fue posible cargar modelo HuggingFace '{self.model_name}' ({e}). "
                "Activando motor léxico/heurístico optimizado para español."
            )
            self.pipeline = None
            self.is_loaded = False

    @staticmethod
    def _clean_text(text: Any) -> str:
        if text is None:
            return ""
        s = str(text).strip()
        s = re.sub(r"\s+", " ", s)
        return s

    def _is_non_informative(self, text: str) -> bool:
        normalized = text.lower().strip()
        normalized_clean = re.sub(r"^[^\w]+|[^\w]+$", "", normalized)
        if len(normalized) <= 1:
            return True
        if normalized in NON_INFORMATIVE_PATTERNS or normalized_clean in NON_INFORMATIVE_PATTERNS:
            return True
        return False

    def _classify_with_lexicon(self, text: str) -> Tuple[str, float]:
        normalized = text.lower()
        
        pos_score = sum(1 for word in POSITIVE_WORDS if re.search(r"\b" + re.escape(word) + r"\b", normalized))
        neg_score = sum(1 for word in NEGATIVE_WORDS if re.search(r"\b" + re.escape(word) + r"\b", normalized))
        
        # Considerar negaciones
        if re.search(r"\bno\s+(es|fue|parece|resulto|resultó)\s+(bueno|buena|excelente|amable)\b", normalized):
            neg_score += 2
            pos_score = max(0, pos_score - 1)
        if re.search(r"\bno\s+(es|fue|parece)\s+(malo|mala|pesimo|pésimo|pesima|pésima)\b", normalized):
            pos_score += 1
            neg_score = max(0, neg_score - 1)

        total = pos_score + neg_score
        if total == 0:
            return "Ninguna Observación", 0.70

        if pos_score > neg_score:
            conf = min(0.99, 0.65 + (pos_score - neg_score) * 0.1)
            return "Positiva", round(conf, 4)
        elif neg_score > pos_score:
            conf = min(0.99, 0.65 + (neg_score - pos_score) * 0.1)
            return "Por mejorar", round(conf, 4)
        else:
            return "Ninguna Observación", 0.60

    def predict_one(self, text: Any) -> Tuple[str, float]:
        cleaned = self._clean_text(text)
        
        if self._is_non_informative(cleaned):
            return "Ninguna Observación", 1.0

        if self.pipeline:
            try:
                results = self.pipeline(cleaned[:512])[0]
                return self._map_pipeline_output(results)
            except Exception as e:
                logger.error(f"Error infiriendo con pipeline: {e}. Usando clasificador léxico.")

        return self._classify_with_lexicon(cleaned)

    def _map_pipeline_output(self, scores_list: List[Dict[str, Any]]) -> Tuple[str, float]:
        score_dict = {item['label'].upper(): item['score'] for item in scores_list}
        
        pos = score_dict.get('POS', score_dict.get('POSITIVE', 0.0))
        neg = score_dict.get('NEG', score_dict.get('NEGATIVE', 0.0))
        neu = score_dict.get('NEU', score_dict.get('NEUTRAL', 0.0))

        dominant_label = max(score_dict, key=score_dict.get)
        dominant_score = score_dict[dominant_label]

        if dominant_score < settings.CONFIDENCE_THRESHOLD:
            return "Ninguna Observación", round(dominant_score, 4)

        if dominant_label in ('POS', 'POSITIVE'):
            return "Positiva", round(pos, 4)
        elif dominant_label in ('NEG', 'NEGATIVE'):
            return "Por mejorar", round(neg, 4)
        else:
            return "Ninguna Observación", round(neu if neu > 0 else dominant_score, 4)

    def predict_batch(self, texts: List[Any]) -> List[Tuple[str, float]]:
        cleaned_texts = [self._clean_text(t) for t in texts]
        results = [None] * len(cleaned_texts)
        texts_to_infer = []
        indices_to_infer = []

        for idx, text in enumerate(cleaned_texts):
            if self._is_non_informative(text):
                results[idx] = ("Ninguna Observación", 1.0)
            else:
                texts_to_infer.append(text[:512])
                indices_to_infer.append(idx)

        if texts_to_infer and self.pipeline:
            try:
                pipeline_outputs = self.pipeline(texts_to_infer)
                for real_idx, out in zip(indices_to_infer, pipeline_outputs):
                    results[real_idx] = self._map_pipeline_output(out)
            except Exception as e:
                logger.error(f"Error procesando batch en pipeline: {e}. Usando clasificador léxico.")
                for real_idx, text in zip(indices_to_infer, texts_to_infer):
                    results[real_idx] = self._classify_with_lexicon(text)
        elif texts_to_infer:
            for real_idx, text in zip(indices_to_infer, texts_to_infer):
                results[real_idx] = self._classify_with_lexicon(text)

        return results


nlp_service = NLPClassifierService()
