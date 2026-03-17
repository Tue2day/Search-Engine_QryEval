"""
Define and store data for the BM25 Retrieval Model.
"""

# Copyright (c) 2026, Carnegie Mellon University.  All Rights Reserved.

from RetrievalModel import RetrievalModel

class RetrievalModelBM25(RetrievalModel):
    """
    Define and store data for the BM25 Retrieval Model.
    """


    # -------------- Methods (alphabetical) ---------------- #

    def __init__(self, k1=1.2, b=0.75):
        self.k1 = float(k1)
        self.b = float(b)
        self.defaultQrySop = '#SUM'
