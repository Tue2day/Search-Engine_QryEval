"""
Access and manage a feature-based learning-to-rank (Ltr) reranker.
"""

# Copyright (c) 2026, Carnegie Mellon University.  All Rights Reserved.

import os
import math
import subprocess

import PyLu				# Used to access RankLib
import Util				# Used to read and write files

from Idx import Idx
from QryParser import QryParser		# Parse queries

class RerankWithLtr:
    """
    Access and manage a feature-based learning-to-rank (Ltr) reranker.
    """

    # -------------- Methods (alphabetical) ---------------- #

    def __init__(self, parameters):
        self._all_feature_ids = list(range(1, 21))
        self._parameters = parameters
        self._toolkit = parameters.get('ltr:toolkit', 'RankLib')
        self._feature_disable = self._parse_disabled_features(
            parameters.get('ltr:featureDisable', ''))
        self._training_query_file = parameters.get('ltr:trainingQueryFile')
        self._training_qrels_file = parameters.get('ltr:trainingQrelsFile')
        self._training_feature_vectors_file = parameters.get(
            'ltr:trainingFeatureVectorsFile')
        self._testing_feature_vectors_file = parameters.get(
            'ltr:testingFeatureVectorsFile')
        self._testing_document_scores = parameters.get(
            'ltr:testingDocumentScores')
        self._model_file = parameters.get('ltr:modelFile')
        self._ranklib_model = parameters.get('ltr:RankLib:model')
        self._ranklib_metric2t = parameters.get('ltr:RankLib:metric2t')
        self._svm_rank_learn_path = parameters.get('ltr:svmRankLearnPath')
        self._svm_rank_classify_path = parameters.get(
            'ltr:svmRankClassifyPath')
        self._svm_rank_param_c = float(parameters.get(
            'ltr:svmRankParamC', 0.001))
        self._bm25_k1 = float(parameters.get('ltr:BM25:k_1', 1.2))
        self._bm25_b = float(parameters.get('ltr:BM25:b', 0.75))
        self._ql_mu = float(parameters.get('ltr:QL:mu', 2500.0))

        # Build the training data and model up front for later reranking.
        self._generate_training_feature_vectors()
        self._train_model()
        

    def rerank(self, batch):
        """
        Update the results for a set of queries with new scores.

        batch: A dict of {qid: {'qstring': qstring,
                                'ranking': [(score, externalId) ...]}
                          ... }
        """
        self._generate_testing_feature_vectors(batch)
        self._score_testing_documents()
        self._apply_reranked_scores(batch)
        return(batch)


    def _build_feature_dict(self, docid, query_terms):
        features = {}
        unique_query_terms = list(dict.fromkeys(query_terms))
        if 1 not in self._feature_disable:
            spam_score = Idx.getAttribute('spamScore', docid)
            if spam_score is not None:
                features[1] = float(spam_score)

        raw_url = Idx.getAttribute('rawUrl', docid)
        if 2 not in self._feature_disable and raw_url is not None:
            features[2] = float(raw_url.count('/'))
        if 3 not in self._feature_disable and raw_url is not None:
            features[3] = 1.0 if 'wikipedia.org' in raw_url.lower() else 0.0

        if 4 not in self._feature_disable:
            page_rank = Idx.getAttribute('PageRank', docid)
            if page_rank is not None:
                features[4] = float(page_rank)

        overlap_fields = {
            7: 'body',
            10: 'title',
            13: 'url',
            16: 'inlink'
        }
        for fid, field in overlap_fields.items():
            if fid in self._feature_disable:
                continue
            overlap = self._feature_overlap(docid, field, unique_query_terms)
            if overlap is not None:
                features[fid] = float(overlap)

        bm25_fields = {
            5: 'body',
            8: 'title',
            11: 'url',
            14: 'inlink'
        }
        for fid, field in bm25_fields.items():
            if fid in self._feature_disable:
                continue
            score = self._feature_bm25(docid, field, unique_query_terms)
            if score is not None:
                features[fid] = float(score)

        ql_fields = {
            6: 'body',
            9: 'title',
            12: 'url',
            15: 'inlink'
        }
        for fid, field in ql_fields.items():
            if fid in self._feature_disable:
                continue
            score = self._feature_ql(docid, field, unique_query_terms)
            if score is not None:
                features[fid] = float(score)

        if 17 not in self._feature_disable:
            score = self._feature_query_term_density(
                docid, 'body', unique_query_terms)
            if score is not None:
                features[17] = float(score)

        if 18 not in self._feature_disable:
            score = self._feature_adjacent_pair_match(
                docid, 'body', query_terms)
            if score is not None:
                features[18] = float(score)

        if 19 not in self._feature_disable:
            score = self._feature_cross_field_agreement(features)
            if score is not None:
                features[19] = float(score)

        if 20 not in self._feature_disable:
            score = self._feature_early_lead_match(
                docid, 'body', unique_query_terms, 100)
            if score is not None:
                features[20] = float(score)

        return(features)


    def _ensure_parent_dir(self, path):
        if path is None:
            return
        parent = os.path.dirname(path)
        if parent != '':
            os.makedirs(parent, exist_ok=True)


    def _format_feature_value(self, value):
        return str(float(value))


    def _format_feature_vector(self, vector):
        feature_ids = self._feature_ids_for_output(vector['features'])
        features = ' '.join(
            f'{fid}:{self._format_feature_value(vector["features"].get(fid, 0.0))}'
            for fid in feature_ids)
        if features != '':
            features = ' ' + features
        return(f'{vector["label"]} qid:{vector["qid"]}{features}  '
               f'# {vector["externalId"]}')


    def _generate_feature_vector(self, qid, query_string, external_id, label):
        query_terms = QryParser.tokenizeString(query_string)
        internal_id = Idx.getInternalDocid(external_id)
        if internal_id is None or internal_id < 0:
            raise Exception(f'Error: Unknown document id {external_id}.')

        return({
            'qid': qid,
            'label': int(label),
            'externalId': external_id,
            'features': self._build_feature_dict(internal_id, query_terms)
        })


    def _feature_overlap(self, docid, field, query_terms):
        term_vector = self._get_nonempty_term_vector(docid, field)
        if term_vector is None:
            return(None)

        return(sum(
            1 for term in query_terms
            if term_vector.indexOfStem(term) != -1
        ))


    def _feature_bm25(self, docid, field, query_terms):
        term_vector = self._get_nonempty_term_vector(docid, field)
        if term_vector is None:
            return(None)

        doc_count = Idx.getDocCount(field)
        if doc_count <= 0:
            return(0.0)

        avgdl = Idx.getSumOfFieldLengths(field) / float(doc_count)
        if avgdl <= 0.0:
            return(0.0)

        doc_len = Idx.getFieldLength(field, docid)
        score = 0.0

        for term in query_terms:
            stem_index = term_vector.indexOfStem(term)
            if stem_index == -1:
                continue

            tf = term_vector.stemFreq(stem_index)
            df = Idx.getDocFreq(field, term)
            idf = math.log((Idx.getNumDocs() + 1.0) / (df + 0.5))
            tf_weight = tf / (
                tf + self._bm25_k1 *
                (1.0 - self._bm25_b + self._bm25_b * doc_len / avgdl))
            score += idf * tf_weight

        return(score)


    def _feature_ql(self, docid, field, query_terms):
        term_vector = self._get_nonempty_term_vector(docid, field)
        if term_vector is None:
            return(None)

        if len(query_terms) == 0:
            return(0.0)

        doc_len = Idx.getFieldLength(field, docid)
        collection_len = Idx.getSumOfFieldLengths(field)
        if doc_len <= 0 or collection_len <= 0:
            return(None)

        log_prob_sum = 0.0
        matched_terms = 0
        for term in query_terms:
            stem_index = term_vector.indexOfStem(term)
            tf = 0 if stem_index == -1 else term_vector.stemFreq(stem_index)
            if tf > 0:
                matched_terms += 1
            ctf = Idx.getTotalTermFreq(field, term)
            mle = ctf / float(collection_len)
            term_prob = (tf + self._ql_mu * mle) / (doc_len + self._ql_mu)

            if term_prob <= 0.0:
                return(0.0)

            log_prob_sum += math.log(term_prob)

        if matched_terms == 0:
            return(0.0)

        return(math.exp(log_prob_sum / len(query_terms)))


    def _feature_cross_field_agreement(self, features):
        bm25_feature_ids = [5, 8, 11, 14]
        matched_fields = sum(
            1 for fid in bm25_feature_ids
            if features.get(fid, 0.0) > 0.0)
        return(matched_fields / float(len(bm25_feature_ids)))


    def _feature_adjacent_pair_match(self, docid, field, query_terms):
        term_vector = self._get_nonempty_term_vector(docid, field)
        if term_vector is None:
            return(None)

        if len(query_terms) < 2:
            return(0.0)

        positions_by_term = self._positions_by_term(term_vector, set(query_terms))
        matched_pairs = 0
        total_pairs = len(query_terms) - 1

        for i in range(total_pairs):
            first_term = query_terms[i]
            second_term = query_terms[i + 1]
            first_positions = positions_by_term.get(first_term, [])
            second_positions = positions_by_term.get(second_term, [])

            if self._has_adjacent_match(first_positions, second_positions):
                matched_pairs += 1

        return(matched_pairs / float(total_pairs))


    def _feature_early_lead_match(self, docid, field, query_terms, cutoff):
        term_vector = self._get_nonempty_term_vector(docid, field)
        if term_vector is None:
            return(None)

        query_term_set = set(query_terms)
        if len(query_term_set) == 0:
            return(0.0)

        lead_terms = set()
        lead_length = min(term_vector.positionsLength(), cutoff)
        for position_i in range(lead_length):
            stem_index = term_vector.stemAt(position_i)
            stem = term_vector.stemString(stem_index)
            if stem is not None:
                lead_terms.add(str(stem))

        return(sum(1 for term in query_term_set if term in lead_terms))


    def _feature_query_term_density(self, docid, field, query_terms):
        term_vector = self._get_nonempty_term_vector(docid, field)
        if term_vector is None:
            return(None)

        doc_len = Idx.getFieldLength(field, docid)
        if doc_len <= 0:
            return(None)

        total_tf = 0.0
        for term in query_terms:
            stem_index = term_vector.indexOfStem(term)
            if stem_index == -1:
                continue
            total_tf += term_vector.stemFreq(stem_index)

        return(total_tf / float(doc_len))


    def _get_nonempty_term_vector(self, docid, field):
        term_vector = Idx.getTermVector(docid, field)
        if (term_vector is None or
            term_vector.stemsLength() == 0 or
            term_vector.positionsLength() == 0):
            return(None)
        return(term_vector)


    def _has_adjacent_match(self, first_positions, second_positions):
        i = 0
        j = 0

        while i < len(first_positions) and j < len(second_positions):
            if second_positions[j] == first_positions[i] + 1:
                return(True)
            if second_positions[j] <= first_positions[i]:
                j += 1
            else:
                i += 1

        return(False)


    def _positions_by_term(self, term_vector, query_terms):
        positions = {term: [] for term in query_terms}
        if len(positions) == 0:
            return(positions)

        for position_i in range(term_vector.positionsLength()):
            stem_index = term_vector.stemAt(position_i)
            stem = term_vector.stemString(stem_index)
            if stem is None:
                continue
            stem = str(stem)
            if stem in positions:
                positions[stem].append(position_i)

        return(positions)


    def _generate_feature_vectors_for_ranking(self, qid, query_string, ranking):
        vectors = []
        for _, external_id in ranking:
            vectors.append(self._generate_feature_vector(
                qid, query_string, external_id, 0))
        return(vectors)


    def _generate_testing_feature_vectors(self, batch):
        if self._testing_feature_vectors_file is None:
            return

        vectors = []
        for qid in sorted(batch.keys(), key=int):
            qstring = QryParser.bowQuery(batch[qid]['qstring'])
            q_vectors = self._generate_feature_vectors_for_ranking(
                qid, qstring, batch[qid]['ranking'])
            vectors.extend(self._normalize_vectors_for_query(q_vectors))

        self._write_feature_vectors(self._testing_feature_vectors_file, vectors)


    def _generate_training_feature_vectors(self):
        if (self._training_query_file is None or
            self._training_qrels_file is None or
            self._training_feature_vectors_file is None):
            return

        queries = Util.read_queries(self._training_query_file)
        qrels = Util.read_qrels(self._training_qrels_file)
        vectors = []
        qrels_by_qid = {}
        for qid, _, external_id, rel in qrels:
            if qid not in queries:
                continue
            qrels_by_qid.setdefault(qid, []).append((external_id, rel))

        for qid in sorted(qrels_by_qid.keys(), key=int):
            q_vectors = []
            for external_id, rel in qrels_by_qid[qid]:
                q_vectors.append(self._generate_feature_vector(
                    qid,
                    queries[qid],
                    external_id,
                    self._normalize_qrels_label(rel)))
            vectors.extend(self._normalize_vectors_for_query(q_vectors))

        self._write_feature_vectors(self._training_feature_vectors_file, vectors)


    def _feature_ids_for_output(self, features):
        if self._toolkit.lower() == 'ranklib':
            return([
                fid for fid in self._all_feature_ids
                if fid not in self._feature_disable
            ])
        return(sorted(features.keys()))


    def _read_document_scores(self):
        if self._testing_document_scores is None:
            return([])

        lines = Util.file_read_strings(self._testing_document_scores)
        if lines is None:
            return([])

        scores = []
        if self._toolkit.lower() == 'ranklib':
            for line in lines:
                parts = line.split('\t')
                if len(parts) < 3:
                    continue
                scores.append(float(parts[2]))
        else:
            for line in lines:
                stripped = line.strip()
                if stripped == '':
                    continue
                scores.append(float(stripped))

        return(scores)


    def _apply_reranked_scores(self, batch):
        scores = self._read_document_scores()
        score_index = 0

        for qid in sorted(batch.keys(), key=int):
            old_ranking = batch[qid]['ranking']
            reranked = []

            for _, external_id in old_ranking:
                if score_index >= len(scores):
                    raise Exception(
                        'Error: Not enough document scores to rerank results.')
                reranked.append((scores[score_index], external_id))
                score_index += 1

            reranked.sort(key=lambda pair: (-pair[0], pair[1]))
            batch[qid]['ranking'] = reranked

        if score_index != len(scores):
            raise Exception(
                'Error: Extra document scores remain after reranking results.')


    def _run_command(self, args):
        try:
            return(subprocess.check_output(
                args,
                stderr=subprocess.STDOUT).decode('UTF-8'))
        except subprocess.CalledProcessError as e:
            output = e.output.decode('UTF-8', errors='replace')
            raise Exception(
                f'Error running command {" ".join(args)}\n{output}') from e


    def _score_testing_documents(self):
        if (self._testing_feature_vectors_file is None or
            self._testing_document_scores is None or
            self._model_file is None):
            return

        self._ensure_parent_dir(self._testing_document_scores)
        toolkit = self._toolkit.lower()

        if toolkit == 'ranklib':
            args = [
                '-rank', self._testing_feature_vectors_file,
                '-load', self._model_file,
                '-score', self._testing_document_scores
            ]
            PyLu.RankLib.main(args)
        elif toolkit == 'svmrank':
            if self._svm_rank_classify_path is None:
                raise Exception(
                    'Error: Missing parameter ltr:svmRankClassifyPath.')
            self._run_command([
                self._svm_rank_classify_path,
                self._testing_feature_vectors_file,
                self._model_file,
                self._testing_document_scores
            ])
        else:
            raise Exception(f'Error: Unknown ltr toolkit {self._toolkit}.')


    def _train_model(self):
        if (self._training_feature_vectors_file is None or
            self._model_file is None):
            return

        self._ensure_parent_dir(self._model_file)
        toolkit = self._toolkit.lower()

        if toolkit == 'ranklib':
            args = [
                '-train', self._training_feature_vectors_file,
                '-ranker', str(self._ranklib_model),
                '-save', self._model_file
            ]
            if self._ranklib_metric2t is not None:
                args.extend(['-metric2t', str(self._ranklib_metric2t)])
            elif str(self._ranklib_model) == '4':
                args.extend(['-metric2t', 'MAP'])
            PyLu.RankLib.main(args)
        elif toolkit == 'svmrank':
            if self._svm_rank_learn_path is None:
                raise Exception('Error: Missing parameter ltr:svmRankLearnPath.')
            self._run_command([
                self._svm_rank_learn_path,
                '-c', str(self._svm_rank_param_c),
                self._training_feature_vectors_file,
                self._model_file
            ])
        else:
            raise Exception(f'Error: Unknown ltr toolkit {self._toolkit}.')


    def _normalize_vectors_for_query(self, vectors):
        if len(vectors) == 0:
            return(vectors)

        if self._toolkit.lower() != 'svmrank':
            return(vectors)

        stats = {}
        for fid in self._all_feature_ids:
            if fid in self._feature_disable:
                continue
            values = [
                vector['features'][fid] for vector in vectors
                if fid in vector['features']
            ]
            if len(values) == 0:
                continue
            stats[fid] = (min(values), max(values))

        normalized_vectors = []
        for vector in vectors:
            normalized = dict(vector)
            normalized_features = {}
            for fid in self._all_feature_ids:
                if fid in self._feature_disable:
                    continue

                if fid not in vector['features']:
                    normalized_features[fid] = 0.0
                    continue

                if fid not in stats:
                    normalized_features[fid] = 0.0
                    continue

                min_value, max_value = stats[fid]
                if max_value == min_value:
                    normalized_features[fid] = 0.0
                else:
                    normalized_features[fid] = (
                        (vector['features'][fid] - min_value) /
                        (max_value - min_value))

            normalized['features'] = normalized_features
            normalized_vectors.append(normalized)

        return(normalized_vectors)


    def _normalize_qrels_label(self, rel):
        rel = int(rel)
        return(0 if rel < 0 else rel)


    def _parse_disabled_features(self, feature_disable):
        if feature_disable in [None, '']:
            return(set())
        if isinstance(feature_disable, list):
            return({int(fid) for fid in feature_disable})
        return({
            int(fid.strip()) for fid in str(feature_disable).split(',')
            if fid.strip() != ''
        })


    def _write_feature_vectors(self, path, vectors):
        self._ensure_parent_dir(path)
        lines = [self._format_feature_vector(vector) for vector in vectors]
        Util.file_write_strings(path, lines)
