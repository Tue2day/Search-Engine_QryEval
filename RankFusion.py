"""
Fuse an existing ranking with a secondary ranking file.
"""

# Copyright (c) 2026, Carnegie Mellon University.  All Rights Reserved.

import Util


class RankFusion:
    """
    Merge the current ranking with a secondary ranking.
    Supported methods:
      - union: keep the current ranking order, then append unseen docs
               from the secondary ranking.
      - interleave: alternate between the current ranking and the
                    secondary ranking while skipping duplicates.
    """

    def __init__(self, parameters):
        self.replace_ranking = True
        self._secondary_path = parameters.get('rankfusion:secondaryRankPath')
        self._method = str(parameters.get('rankfusion:method', 'interleave')).lower()
        self._max_input_rank = parameters.get('rankfusion:maxInputRank')
        self._max_output_rank = int(parameters.get('rankfusion:maxOutputRank', 1000))

        if self._secondary_path is None:
            raise Exception(
                'Error: Missing parameter rankfusion:secondaryRankPath.')

        if self._method not in ['union', 'interleave']:
            raise Exception(
                f'Error: Unknown rankfusion:method {self._method}.')

        self._secondary_rankings = Util.read_rankings(
            self._secondary_path, self._max_input_rank)

    def rerank(self, batch):
        for qid in batch:
            primary = batch[qid].get('ranking', [])
            secondary = self._secondary_rankings.get(qid, [])
            merged = self._merge_rankings(primary, secondary)
            batch[qid]['ranking'] = self._assign_scores(merged)

        return(batch)

    def _assign_scores(self, docids):
        ranking = []
        total = len(docids)
        for i, external_id in enumerate(docids):
            ranking.append((float(total - i), external_id))
        return(ranking)

    def _merge_rankings(self, primary, secondary):
        primary_ids = [eid for _, eid in primary]
        secondary_ids = [eid for _, eid in secondary]

        if self._method == 'union':
            merged = self._union(primary_ids, secondary_ids)
        elif self._method == 'interleave':
            merged = self._interleave(primary_ids, secondary_ids)
        else:
            raise Exception(f'Error: Unsupported method {self._method}.')

        return(merged[:self._max_output_rank])

    def _interleave(self, primary_ids, secondary_ids):
        merged = []
        seen = set()
        i = 0
        j = 0

        while i < len(primary_ids) or j < len(secondary_ids):
            if i < len(primary_ids):
                docid = primary_ids[i]
                i += 1
                if docid not in seen:
                    merged.append(docid)
                    seen.add(docid)

            if j < len(secondary_ids):
                docid = secondary_ids[j]
                j += 1
                if docid not in seen:
                    merged.append(docid)
                    seen.add(docid)

        return(merged)

    def _union(self, primary_ids, secondary_ids):
        merged = []
        seen = set()

        for docid in primary_ids + secondary_ids:
            if docid not in seen:
                merged.append(docid)
                seen.add(docid)

        return(merged)
