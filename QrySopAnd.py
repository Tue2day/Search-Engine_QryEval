"""
The AND operator for all retrieval models.
"""

# Copyright (c) 2026, Carnegie Mellon University.  All Rights Reserved.

import re
import sys

from QrySop import QrySop
from RetrievalModelUnrankedBoolean import RetrievalModelUnrankedBoolean
from RetrievalModelRankedBoolean import RetrievalModelRankedBoolean


class QrySopAnd(QrySop):
    """
    The And operator for all retrieval models.
    """

    # -------------- Methods (alphabetical) ---------------- #


    def __init__(self):
        QrySop.__init__(self)		# Inherit from QrySop


    def docIteratorHasMatch(self, r):
        """
        Indicates whether the query has a match.

        r: The retrieval model that determines what is a match.
        Returns True if the query matches, otherwise False.
        """
        return self.docIteratorHasMatchAll(r)


    def getScore(self, retrievalModel):
        """
        Get a score for the document that docIteratorHasMatch matched.

        retrievalModel: retrieval model parameters

        Returns the document score.

        throws IOException: Error accessing the Lucene index
        """

        if isinstance(retrievalModel, RetrievalModelUnrankedBoolean):
            return self.__getScoreUnRankedBoolean(retrievalModel)
        elif isinstance(retrievalModel, RetrievalModelRankedBoolean):
            return self.__getScoreRankedBoolean(retrievalModel)
        else:
            raise Exception('{}.{} does not support {}'.format(
                self.__class__.__name__,
                sys._getframe().f_code.co_name,
                retrievalModel.__class__.__name__))


    def __getScoreUnRankedBoolean(self, r):
        """
        getScore for UnRanked Boolean retrieval models.

        r: The retrieval model that determines how scores are calculated.
        Returns the document score.
        throws IOException: Error accessing the Lucene index
        """

        # In Unranked Boolean AND, if a document matches,
        # its score is always 1.0
        # No need to check each query argument
        return 1.0
    
    def __getScoreRankedBoolean(self, r):
        """
        getScore for Ranked Boolean retrieval models.

        r: The retrieval model that determines how scores are calculated.
        Returns the document score.
        throws IOException: Error accessing the Lucene index
        """

        # Ranked Boolean AND uses the minimum score among its arguments
        score = float('inf')
        docid = self.docIteratorGetMatch()

        for q_i in self._args:
            if (q_i.docIteratorHasMatch(r) 
                and q_i.docIteratorGetMatch() == docid):
                score = min(score, q_i.getScore(r))

        return score