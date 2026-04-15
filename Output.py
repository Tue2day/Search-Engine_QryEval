"""
Write results in various formats.
"""

# Copyright (c) 2026, Carnegie Mellon University.  All Rights Reserved.

from TeIn import TeIn
from RagSupport import write_qain_file

class Output:

    # -------------- Methods (alphabetical) ---------------- #

    def __init__(self, parameters):
        self._type = parameters['type']
        self._outputPath = parameters['outputPath']
        self._outputLength = parameters.get('outputLength', 1000)


    def close(self):
         if '_teIn' in vars(self):
             self._teIn.close()


    def execute(self, batch):
        """
        Write the output about the batch to a file

        batch: A dict of {qid: {'qstring': qstring,
                                'ranking': [(score, externalId) ...]}
                          ... }
        """
        if self._type == 'trec_eval':
            teIn = TeIn(self._outputPath, self._outputLength)
            for qid in batch:
                teIn.appendQuery(qid, batch[qid]['ranking'], 'reference')
            teIn.close()
        elif self._type == 'triviaqa_evaluation':
            answers = {}
            for qid in batch:
                answers[qid] = batch[qid].get('answer', '')
            write_qain_file(self._outputPath, answers)
        else:
            raise Exception('Error: Unknown Output format')
            
        return(batch)
