import math
from functools import cache
from Idx import Idx


class RewriteWithPrf:
    """
    Implements Pseudo Relevance Feedback algorithms.
    """

    # Cached Index Access 

    @staticmethod
    @cache
    def _cached_doc_freq(field, term):
        return Idx.getDocFreq(field, term)

    @staticmethod
    @cache
    def _cached_total_term_freq(field, term):
        return Idx.getTotalTermFreq(field, term)

    @staticmethod
    @cache
    def _cached_doc_count():
        return Idx.getNumDocs()

    @staticmethod
    @cache
    def _cached_sum_field_lengths(field):
        return Idx.getSumOfFieldLengths(field)

    @staticmethod
    @cache
    def _cached_okapi_rsj(field, term):
        N = RewriteWithPrf._cached_doc_count()
        df = RewriteWithPrf._cached_doc_freq(field, term)

        if df <= 0:
            return 0.0

        numerator = N - df + 0.5
        denominator = df + 0.5

        if denominator <= 0:
            return 0.0

        return max(0.0, math.log(numerator / denominator))

    @staticmethod
    @cache
    def _cached_rm3_idf(field, term):
        ctf = RewriteWithPrf._cached_total_term_freq(field, term)
        collection_len = RewriteWithPrf._cached_sum_field_lengths(field)

        if ctf <= 0 or collection_len <= 0:
            return 0.0

        p_t_c = ctf / float(collection_len)
        return math.log(1.0 / p_t_c)

    def __init__(self, parameters):
        self.algorithm = parameters.get('prf:algorithm', 'okapi').lower()
        self.num_docs = int(parameters.get('prf:numDocs', 0))
        self.num_terms = int(parameters.get('prf:numTerms', 0))
        self.field_in = parameters.get('prf:expansionFieldIn', 'body')
        self.field_out = parameters.get('prf:expansionFieldOut', 'body')

        self.rm3_orig_weight = float(parameters.get('prf:rm3:origWeight', 0.0))

    def rewrite(self, ranking, original_query):
        """
        Perform PRF (Okapi or RM3).
        """
        k = min(len(ranking), self.num_docs)
        top_docs = ranking[:k]

        if self.algorithm == 'rm3':
            return self._rewrite_rm3(top_docs, original_query)
        else:
            return self._rewrite_okapi(top_docs, original_query)


    # OKAPI PRF

    def _rewrite_okapi(self, top_docs, original_query):
        """
        Perform PRF and return the new query string and learned terms.
        """

        term_scores = self._compute_scores_okapi(top_docs)

        # Sort by score descending, lexical ascending
        all_sorted = sorted(term_scores.items(), key=lambda x: (-x[1], x[0]))
        top_k_terms = all_sorted[:self.num_terms]

        # Output order: least important -> most important
        final_sorted_terms = sorted(top_k_terms, key=lambda x: (x[1], x[0]))

        # Format for .qryOut
        if self.field_out == 'body':
            file_terms = [t[0] for t in final_sorted_terms]
        else:
            file_terms = [f"{t[0]}.{self.field_out}" for t in final_sorted_terms]

        file_output_str = f"#SUM( {' '.join(file_terms)} )"

        # Internal expanded query
        query_terms = [f"{t[0]}.{self.field_out}" for t in final_sorted_terms]
        learned_part = f"#SUM( {' '.join(query_terms)} )"

        if self.rm3_orig_weight > 0.0:
            weight_learned = 1.0 - self.rm3_orig_weight
            new_query_str = (f"#WSUM( {self.rm3_orig_weight} #SUM({original_query}) "
                             f"{weight_learned} {learned_part} )")
        else:
            new_query_str = learned_part

        return new_query_str, file_output_str


    # RM3 PRF

    def _rewrite_rm3(self, top_docs, original_query):
        """
        RM3 PRF Logic.
        """

        term_scores = self._compute_scores_rm3(top_docs)

        all_sorted = sorted(term_scores.items(), key=lambda x: (-x[1], x[0]))
        top_k_terms = all_sorted[:self.num_terms]

        # Normalize scores to probabilities
        total_score = sum(score for term, score in top_k_terms)
        if total_score == 0: total_score = 1.0

        normalized_terms = []
        for term, score in top_k_terms:
            normalized_terms.append((term, score / total_score))

        final_sorted_terms = sorted(normalized_terms, key=lambda x: (x[1], x[0]))

        # Format for .qryOut
        file_args = []
        for term, prob in final_sorted_terms:
            if self.field_out == 'body':
                file_args.append(f"{prob:.20f} {term}")
            else:
                file_args.append(f"{prob:.20f} {term}.{self.field_out}")

        file_output_str = f"#WSUM( {' '.join(file_args)} )"

        # Internal learned query
        query_args = [f"{prob:.20f} {term}.{self.field_out}"
                      for term, prob in final_sorted_terms]

        learned_part = f"#WSUM( {' '.join(query_args)} )"

        weight_learned = 1.0 - self.rm3_orig_weight

        if self.rm3_orig_weight > 0.0:
            new_query_str = (f"#WSUM( {self.rm3_orig_weight} #SUM({original_query}) "
                             f"{weight_learned} {learned_part} )")
        else:
            new_query_str = learned_part

        return new_query_str, file_output_str


    # SCORE COMPUTATION

    def _compute_scores_okapi(self, top_docs):
        """
        Compute Okapi scores for terms in the top documents.
        Score(t) = rdf(t) * RSJ(t)
        """

        doc_term_sets = []

        # Collect unique term sets for each document
        for _, ext_id in top_docs:
            int_id = Idx.getInternalDocid(ext_id)
            if int_id == -1:
                continue

            tv = Idx.getTermVector(int_id, self.field_in)
            if tv is None or tv.stemsLength() == 0:
                continue

            term_set = set()

            for i in range(1, tv.stemsLength()):
                term = tv.stemString(i)
                if term is None:
                    continue
                if '.' in term or ',' in term or not term.isascii():
                    continue
                term_set.add(term)

            doc_term_sets.append(term_set)

        vocab = set().union(*doc_term_sets) if doc_term_sets else set()

        scores = {}

        for term in vocab:
            rdf = sum(1 for s in doc_term_sets if term in s)
            rsj = RewriteWithPrf._cached_okapi_rsj(self.field_in, term)
            score = rdf * rsj
            if score > 0.0:
                scores[term] = score

        return scores

    def _compute_scores_rm3(self, top_docs):
        """
        Compute RM3 scores.
        """

        scores = {}

        for doc_score, ext_id in top_docs:
            int_id = Idx.getInternalDocid(ext_id)
            if int_id == -1:
                continue

            tv = Idx.getTermVector(int_id, self.field_in)
            if tv is None or tv.stemsLength() == 0:
                continue

            doc_len = tv.positionsLength()
            if doc_len == 0:
                continue

            for i in range(1, tv.stemsLength()):
                term = tv.stemString(i)
                if term is None:
                    continue
                if '.' in term or ',' in term or not term.isascii():
                    continue

                tf = tv.stemFreq(i)
                p_t_d = tf / float(doc_len)

                idf = RewriteWithPrf._cached_rm3_idf(self.field_in, term)
                weight = p_t_d * doc_score * idf

                if weight > 0.0:
                    scores[term] = scores.get(term, 0.0) + weight

        return scores
