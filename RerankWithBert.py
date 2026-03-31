"""
Access and manage a BERT reranker.
"""

# Copyright (c) 2026, Carnegie Mellon University.  All Rights Reserved.

from Idx import Idx


class RerankWithBert:
    """
    Access and manage a BERT reranker.
    """

    _BERT_MAX_SEQUENCE_LENGTH = 512

    # -------------- Methods (alphabetical) ---------------- #

    def __init__(self, parameters):
        self._parameters = parameters
        self._model_path = parameters.get('bertrr:modelPath')
        self._psg_len = int(parameters.get('bertrr:psgLen', 0))
        self._psg_stride = int(parameters.get('bertrr:psgStride', 0))
        self._psg_cnt = int(parameters.get('bertrr:psgCnt', 1))
        self._max_title_length = int(parameters.get('bertrr:maxTitleLength', 0))
        self._score_aggregation = parameters.get(
            'bertrr:scoreAggregation', 'firstp').lower()

        if self._model_path is None:
            raise Exception('Error: Missing parameter bertrr:modelPath.')
        if self._psg_len < 0:
            raise Exception('Error: bertrr:psgLen must be >= 0.')
        if self._psg_stride < 0:
            raise Exception('Error: bertrr:psgStride must be >= 0.')
        if self._psg_cnt <= 0:
            raise Exception('Error: bertrr:psgCnt must be > 0.')
        if self._score_aggregation not in ['firstp', 'avgp', 'maxp']:
            raise Exception(
                f'Error: Unknown bertrr:scoreAggregation '
                f'{self._score_aggregation}.')

        try:
            import torch
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer)
        except ImportError as e:
            raise Exception(
                'Error: BERT reranking requires torch and transformers '
                'to be installed in the active Python environment.') from e

        self._torch = torch
        self._device = torch.device('cpu')
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self._model_path,
            num_labels=1)
        self._model.to(self._device)
        self._model.eval()


    def rerank(self, batch):
        """
        Update the results for a set of queries with new scores.

        batch: A dict of {qid: {'qstring': qstring,
                                'ranking': [(score, externalId) ...]}
                          ... }
        """
        for qid in sorted(batch.keys(), key=int):
            qstring = batch[qid]['qstring']
            reranked = []

            for _, external_id in batch[qid]['ranking']:
                doc_score = self._score_document(qstring, external_id)
                reranked.append((doc_score, external_id))

            reranked.sort(key=lambda pair: (-pair[0], pair[1]))
            batch[qid]['ranking'] = reranked

        return(batch)


    def _aggregate_scores(self, passage_scores):
        if len(passage_scores) == 0:
            return(float('-inf'))

        if self._score_aggregation == 'firstp':
            return(passage_scores[0])
        if self._score_aggregation == 'avgp':
            return(sum(passage_scores) / float(len(passage_scores)))
        if self._score_aggregation == 'maxp':
            return(max(passage_scores))

        raise Exception(
            f'Error: Unknown score aggregation {self._score_aggregation}.')


    def _build_body_passages(self, body_tokens):
        if len(body_tokens) == 0:
            return([[]])

        if self._psg_len <= 0:
            return([[]])

        passages = []
        start = 0
        prev_end = -1

        while start < len(body_tokens) and len(passages) < self._psg_cnt:
            end = min(start + self._psg_len, len(body_tokens))
            if end <= prev_end:
                break

            passages.append(body_tokens[start:end])
            prev_end = end

            if end >= len(body_tokens):
                break

            if self._psg_stride <= 0:
                break
            start += self._psg_stride

        if len(passages) == 0:
            return([[]])

        return(passages)


    def _build_passage_text(self, title_tokens, body_passage_tokens):
        parts = []

        if self._max_title_length > 0 and len(title_tokens) > 0:
            parts.extend(title_tokens[:self._max_title_length])

        parts.extend(body_passage_tokens)

        return(' '.join(parts).strip())


    def _build_passages(self, title_string, body_string):
        title_tokens = self._tokenize_text(title_string)
        body_tokens = self._tokenize_text(body_string)

        passages = []
        for body_passage_tokens in self._build_body_passages(body_tokens):
            passage_text = self._build_passage_text(
                title_tokens, body_passage_tokens)
            passages.append(passage_text)

        if len(passages) == 0:
            passages = ['']

        return(passages[:self._psg_cnt])


    def _encode_q_psg(self, qstring, passage_string):
        tensors = self._tokenizer.encode_plus(
            [qstring, passage_string],
            add_special_tokens=True,
            max_length=self._BERT_MAX_SEQUENCE_LENGTH,
            truncation='only_second',
            return_tensors='pt')
        return({
            name: tensor.to(self._device)
            for name, tensor in tensors.items()
        })


    def _score_document(self, qstring, external_id):
        internal_id = Idx.getInternalDocid(external_id)
        if internal_id is None or internal_id < 0:
            raise Exception(f'Error: Unknown document id {external_id}.')

        title_string = Idx.getAttribute('title-string', internal_id) or ''
        body_string = Idx.getAttribute('body-string', internal_id) or ''

        passage_scores = []
        for passage_string in self._build_passages(title_string, body_string):
            encoded = self._encode_q_psg(qstring, passage_string)
            passage_scores.append(self._score_sequence(encoded))

        return(self._aggregate_scores(passage_scores))


    def _score_sequence(self, tensors_dict):
        with self._torch.no_grad():
            outputs = self._model(**tensors_dict)
            return(outputs.logits.data.item())


    def _tokenize_text(self, text):
        if text is None:
            return([])
        text = str(text).strip()
        if text == '':
            return([])
        return(text.split())
