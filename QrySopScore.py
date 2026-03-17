"""
The SCORE operator for all retrieval models.
"""

# Copyright (c) 2026, Carnegie Mellon University.  All Rights Reserved.

import math
from functools import cache

from Idx import Idx
from QrySop import QrySop
from QryIop import QryIop
from RetrievalModelUnrankedBoolean import RetrievalModelUnrankedBoolean
from RetrievalModelRankedBoolean import RetrievalModelRankedBoolean
from RetrievalModelBM25 import RetrievalModelBM25


class QrySopScore(QrySop):

    # Cached Index Access

    @staticmethod
    @cache
    def _cached_num_docs():
        return Idx.getNumDocs()

    @staticmethod
    @cache
    def _cached_sum_field_lengths(field):
        return Idx.getSumOfFieldLengths(field)

    @staticmethod
    @cache
    def _cached_doc_count(field):
        return Idx.getDocCount(field)

    @staticmethod
    @cache
    def _cached_idf(field, df):
        N = QrySopScore._cached_num_docs()
        return math.log((N + 1) / (df + 0.5))

    @staticmethod
    @cache
    def _cached_field_length(field, docid):
        return Idx.getFieldLength(field, docid)

    # -------------- Methods (alphabetical) ---------------- #

    def __init__(self):
        QrySop.__init__(self)		# Inherit from QrySop


    def docIteratorHasMatch(self, r):
        """
        Indicates whether the query has a match.
        r: The retrieval model that determines what is a match.

        Returns True if the query matches, otherwise False.
        """
        return(self.docIteratorHasMatchFirst(r))


    def getScore(self, r):
        """
        Get a score for the document that docIteratorHasMatch matched.
        
        r: The retrieval model that determines how scores are calculated.
        Returns the document score.
        throws IOException: Error accessing the Lucene index.
        """

        if isinstance(r, RetrievalModelUnrankedBoolean):
            return self.__getScoreUnrankedBoolean(r)
        elif isinstance(r, RetrievalModelRankedBoolean):
            return self.__getScoreRankedBoolean(r)
        elif isinstance(r, RetrievalModelBM25):
            return self.__getScoreBM25(r)
        else:
            raise Exception(
                '{} does not support the #SCORE operator.'.format(
                    r.__class__.__name__))


    def __getScoreUnrankedBoolean(self, r):
        """
        getScore for the Unranked retrieval model.

        r: The retrieval model that determines how scores are calculated.
        Returns the document score.
        throws IOException: Error accessing the Lucene index.
        """
        if not self.docIteratorHasMatchCache():
            return 0.0
        else:
            return 1.0
        
    def __getScoreRankedBoolean(self, r):
        """
        getScore for the Ranked retrieval model.

        r: The retrieval model that determines how scores are calculated.
        Returns the document score.
        throws IOException: Error accessing the Lucene index.
        """
        if not self.docIteratorHasMatchCache():
            return 0.0
        
        # Get a QryIopXxx operator
        q = self._args[0]
        # Get the posting corresponding to the current matched document
        posting = q.docIteratorGetMatchPosting()
        return posting.tf
    
    def __getScoreBM25(self, r):
        """
        getScore for the Ranked retrieval model.

        r: The retrieval model that determines how scores are calculated.
        Returns the document score.
        throws IOException: Error accessing the Lucene index.
        """
        if not self.docIteratorHasMatchCache():
            return 0.0

        q = self._args[0]
        posting = q.docIteratorGetMatchPosting()

        df = q.getDf()
        field = q._field
        docid = self.docIteratorGetMatch()

        # Cached IDF
        idf = QrySopScore._cached_idf(field, df)

        tf = posting.tf

        # Cached field length
        dl = QrySopScore._cached_field_length(field, docid)

        # Cached average document length
        sum_field_len = QrySopScore._cached_sum_field_lengths(field)
        doc_count = QrySopScore._cached_doc_count(field)
        avgdl = sum_field_len / doc_count if doc_count > 0 else 0.0

        tf_weight = tf / (tf + r.k1 * (1 - r.b + r.b * dl / avgdl))

        score = idf * tf_weight

        return score
 

    def initialize(self, r):
        """
        Initialize the query operator (and its arguments), including any
        internal iterators.  If the query operator is of type QryIop, it
        is fully evaluated, and the results are stored in an internal
        inverted list that may be accessed via the internal iterator.

        r: A retrieval model that guides initialization.
        throws IOException: Error accessing the Lucene index.
        """
        q = self._args[ 0 ]
        q.initialize(r)
