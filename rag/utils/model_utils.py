from sentence_transformers import SentenceTransformer
import os
import logging
import torch
_model = None

def load_embedding_model():
    global _model
    try:
        if _model is None:
            model_name = os.getenv("MODEL_NAME")  # Model adı burada sabitlenmiş
            logging.info(f"Loading embedding model: {model_name}")
            _model = SentenceTransformer(model_name)
            logging.info("Embedding model loaded successfully.")
            if torch.cuda.is_available():
                _model = _model.to('cuda')  # GPU'ya taşıyoruz
                logging.info(f"Model successfully moved to GPU.")
            else:
                logging.info(f"CUDA is not available. Using CPU.")
        return _model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise RuntimeError(f"Error loading model: {e}")


def get_embedding_model():
    if _model is None:
        raise RuntimeError("Embedding model is not loaded!")
    return _model

def unload_embedding_model():
    global _model
    try:
        if _model is not None:
            logging.info("Unloading embedding model...")
            _model = None
        logging.info("Embedding model unloaded successfully.")
    except Exception as e:
        logging.error(f"Error unloading model: {e}")
        raise RuntimeError(f"Error unloading model: {e}")