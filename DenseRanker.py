"""
Dense first-stage retrieval.
"""

# Copyright (c) 2026, Carnegie Mellon University.  All Rights Reserved.

from Idx import Idx

from RagSupport import (
    DEFAULT_DENSE_INDEX_PATH,
    DEFAULT_DENSE_MODEL_PATH,
    get_dense_encoder,
    load_faiss_index)


class DenseRanker:
    """
    Use a FAISS index and dense encoder to produce document rankings.
    """

    def __init__(self, parameters):
        self._index_path = parameters.get(
            'dense:indexPath', DEFAULT_DENSE_INDEX_PATH)
        self._model_path = parameters.get(
            'dense:modelPath', DEFAULT_DENSE_MODEL_PATH)
        self._max_results = int(parameters.get('outputLength', 1000))

        self._encoder = get_dense_encoder(self._model_path)
        self._faiss_index = load_faiss_index(self._index_path)


    def get_rankings(self, batch):
        for qid in batch:
            qstring = batch[qid]['qstring']
            print(f'{qid}: {qstring}', flush=True)
            query_vector = self._encoder.encode(qstring).reshape(1, -1)
            scores, internal_ids = self._faiss_index.search(
                query_vector, self._max_results)

            ranking = []
            for score, internal_id in zip(scores[0], internal_ids[0]):
                internal_id = int(internal_id)
                if internal_id < 0 or internal_id >= Idx.getNumDocs():
                    continue
                external_id = Idx.getExternalDocid(internal_id)
                ranking.append((float(score), external_id))

            batch[qid]['ranking'] = ranking

        return(batch)
