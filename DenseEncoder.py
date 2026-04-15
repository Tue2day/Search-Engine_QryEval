"""
Shared dense text encoding utilities.
"""

# Copyright (c) 2026, Carnegie Mellon University.  All Rights Reserved.

import numpy as np


class DenseEncoder:
    """
    Load a dense encoder model once and reuse it across DPR and RAG.
    """

    _instances = {}
    _MODEL_MAX_SEQUENCE_LENGTH = 512

    def __new__(cls, model_path):
        model_path = str(model_path)
        if model_path in cls._instances:
            return(cls._instances[model_path])

        instance = super().__new__(cls)
        cls._instances[model_path] = instance
        return(instance)


    def __init__(self, model_path):
        if getattr(self, '_initialized', False):
            return

        if model_path is None:
            raise Exception('Error: Dense encoder requires a model path.')

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            raise Exception(
                'Error: Dense retrieval requires torch and transformers '
                'to be installed in the active Python environment.') from e

        self._torch = torch
        self._device = torch.device('cpu')
        self._model_path = str(model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)
        self._model = AutoModel.from_pretrained(self._model_path)
        self._model.to(self._device)
        self._model.eval()
        self._initialized = True


    def encode(self, texts, max_length=None):
        """
        Encode one string or a list of strings into float32 vectors.
        """
        if max_length is None:
            max_length = self._MODEL_MAX_SEQUENCE_LENGTH

        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        cleaned_texts = [self._clean_text(text) for text in texts]
        vectors = []

        for text in cleaned_texts:
            tensors = self._tokenizer.encode_plus(
                text,
                max_length=max_length,
                truncation=True,
                return_tensors='pt')
            tensors = {
                name: tensor.to(self._device)
                for name, tensor in tensors.items()
            }

            with self._torch.no_grad():
                outputs = self._model(**tensors)
                embedding = outputs.last_hidden_state[:, 0]

            vector = embedding.squeeze().detach().cpu().numpy().astype(np.float32)
            vectors.append(vector)

        if single_input:
            return(vectors[0])
        return(np.array(vectors, dtype=np.float32))


    def _clean_text(self, text):
        if text is None:
            return('')
        return(str(text).strip())
