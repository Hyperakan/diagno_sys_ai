from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from models.models import RerankerModel
import os
import logging
import torch

_embedding_model = None
_reranker_model = None

# Embedding Model Utils
def load_embedding_model():
    global _embedding_model
    try:
        if _embedding_model is None:
            model_name = os.getenv("EMBEDDING_MODEL_NAME")  # Model adı burada sabitlenmiş
            if _embedding_model is None:
                model_path = f"/app/embedding_models/{model_name}"  # Yerel model yolunu belirtiyoruz
                if os.path.exists(model_path):
                    _embedding_model = SentenceTransformer(model_path)
                    logging.info("Model loaded from local storage.")
                else:
                    logging.info(f"Installing embedding model: {model_name}")
                    _embedding_model = SentenceTransformer(model_name)
                    _embedding_model.save(f"/app/embedding_models/{model_name}")
                    
            logging.info("Embedding model loaded successfully.")
            if torch.cuda.is_available():
                _embedding_model = _embedding_model.to('cuda')  # GPU'ya taşıyoruz
                logging.info(f"Model successfully moved to GPU.")
            else:
                logging.info(f"CUDA is not available. Using CPU.")
        return _embedding_model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise RuntimeError(f"Error loading model: {e}")


def get_embedding_model():
    if _embedding_model is None:
        raise RuntimeError("Embedding model is not loaded!")
    return _embedding_model

def unload_embedding_model():
    global _embedding_model
    try:
        if _embedding_model is not None:
            logging.info("Unloading embedding model...")
            _embedding_model = None
        logging.info("Embedding model unloaded successfully.")
    except Exception as e:
        logging.error(f"Error unloading model: {e}")
        raise RuntimeError(f"Error unloading model: {e}")

# Reranker Model Utils    
def load_reranker_model():
    model_name = os.getenv("RERANKER_MODEL_NAME")
    global _reranker_model
    try:
        if _reranker_model is None:
            model_path = f"/app/reranker_models/{model_name}/model"
            tokenizer_path = f"/app/reranker_models/{model_name}/tokenizer"
            if os.path.exists(model_path) and os.path.exists(tokenizer_path):
                logging.info("Loading reranker model from local storage.")
                _reranker_model = RerankerModel()
                _reranker_model.model = AutoModelForSequenceClassification.from_pretrained(model_path)
                _reranker_model.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                _reranker_model.use_fp16 = False
                logging.info("Reranker model loaded from local storage.")
            else:
                logging.info(f"Installing reranker model: {model_name}")
                _reranker_model = RerankerModel()
                _reranker_model.model = AutoModelForSequenceClassification.from_pretrained(model_name)
                _reranker_model.tokenizer = AutoTokenizer.from_pretrained(model_name)
                _reranker_model.use_fp16 = False
                _reranker_model.model.save_pretrained(f"/app/reranker_models/{model_name}/model")
                _reranker_model.tokenizer.save_pretrained(f"/app/reranker_models/{model_name}/tokenizer")
            logging.info("Reranker model loaded successfully.")
            if torch.cuda.is_available():
                _reranker_model.model = _reranker_model.model.to('cuda')
                logging.info(f"Model successfully moved to GPU.")
            else:
                logging.info(f"CUDA is not available. Using CPU.")
        return _reranker_model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise RuntimeError(f"Error loading model: {e}")

def get_reranker_model():
    if _reranker_model is None:
        raise RuntimeError("Reranker model is not loaded!")
    return _reranker_model

def unload_reranker_model():
    global _reranker_model
    try:
        if _reranker_model is not None:
            logging.info("Unloading reranker model...")
            _reranker_model = None
        logging.info("Reranker model unloaded successfully.")
    except Exception as e:
        logging.error(f"Error unloading model: {e}")
        raise RuntimeError(f"Error unloading model: {e}")