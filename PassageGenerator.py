"""
Utilities for turning documents into passages.
"""

# Copyright (c) 2026, Carnegie Mellon University.  All Rights Reserved.


class PassageGenerator:
    """
    Build passages from title/body strings using token windows.
    """

    def __init__(self, psg_len, psg_stride, psg_cnt, max_title_length):
        self._psg_len = int(psg_len)
        self._psg_stride = int(psg_stride)
        self._psg_cnt = int(psg_cnt)
        self._max_title_length = int(max_title_length)


    def build_passages(self, title_string, body_string):
        title_tokens = self._tokenize_text(title_string)
        body_tokens = self._tokenize_text(body_string)

        passages = []
        for body_passage_tokens in self._build_body_passages(body_tokens):
            passages.append(self._build_passage_text(
                title_tokens, body_passage_tokens))

        if len(passages) == 0:
            return([''])

        return(passages[:self._psg_cnt])


    def _build_body_passages(self, body_tokens):
        if len(body_tokens) == 0 or self._psg_len <= 0:
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

            if end >= len(body_tokens) or self._psg_stride <= 0:
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


    def _tokenize_text(self, text):
        if text is None:
            return([])
        text = str(text).strip()
        if text == '':
            return([])
        return(text.split())
