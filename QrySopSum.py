"""
The SUM operator for all retrieval models.
"""

# Copyright (c) 2026, Carnegie Mellon University.  All Rights Reserved.

import sys

from QrySop import QrySop
from RetrievalModelBM25 import RetrievalModelBM25


class QrySopSum(QrySop):
    """
    The SUM  operator for all retrieval models.
    """

    # -------------- Methods (alphabetical) ---------------- #


    def __init__( self ):
        QrySop.__init__( self )		# Inherit from QrySop


    # STUDENTS:
    # Add new methods below. See QrySop.py for guidance about
    # the new methods that you need to define.
    def docIteratorHasMatch(self, r):
        return self.docIteratorHasMatchMin(r)
    
    def getScore(self, retrievalModel):
        if isinstance(retrievalModel, RetrievalModelBM25):
            return self.__getScoreBM25(retrievalModel)
        else:
            raise Exception('{}.{} does not support {}'.format(
                self.__class__.__name__,
                sys._getframe().f_code.co_name,
                retrievalModel.__class__.__name__))
        
    def __getScoreBM25(self, r):
        """
        getScore for BM25 retrieval models.

        r: The retrieval model that determines how scores are calculated.
        Returns the document score.
        throws IOException: Error accessing the Lucene index
        """
        
        score = 0.0
        docid = self.docIteratorGetMatch()

        for q_i in self._args:
            if (q_i.docIteratorHasMatch(r) 
                and q_i.docIteratorGetMatch() == docid):
                score += q_i.getScore(r)

        return score
