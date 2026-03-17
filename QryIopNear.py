"""The Near/n operator for all retrieval models."""

# Copyright (c) 2026, Carnegie Mellon University.  All Rights Reserved.

from InvList import InvList
from QryIop import QryIop

class QryIopNear(QryIop):
    """The NEAR/n operator for all retrieval models."""

    # -------------- Methods (alphabetical) ---------------- #

    def __init__(self, distance):
        """Create an empty NEAR/n query node with a given window size."""
        QryIop.__init__(self)		# Inherit from QryIop
        self.distance = distance


    def evaluate(self):
        """"
        Evaluate the query operator; the result is an internal inverted
        list that may be accessed via the internal iterators.

        throws IOException: Error accessing the Lucene index.
        """

        # Create an empty inverted list.
        self.invertedList = InvList(self._field)

        if len(self._args) == 0:	# Should not occur if the
            return			# query optimizer did its job


        while True:

            # Find the maximum current docid among all arguments
            maxDocid = None
            for q_i in self._args:
                # If any of the argument inverted lists are depleted, NEAR is exhausted.
                if not q_i.docIteratorHasMatch(None):
                    return
                q_i_docid = q_i.docIteratorGetMatch()
                if maxDocid is None or q_i_docid > maxDocid:
                    maxDocid = q_i_docid

            # Advance all arguments to maxDocid
            isMatchAll = True
            for q_i in self._args:
                q_i.docIteratorAdvanceTo(maxDocid)
                if (not q_i.docIteratorHasMatch(None) or
                        q_i.docIteratorGetMatch() != maxDocid):
                    isMatchAll = False
                    break
            
            if not isMatchAll:
                continue

            # Check for NEAR condition
            positions_list = []

            for q_i in self._args:
                q_i.locIteratorIndex = 0
            
            while True:
                # Check if all arguments have a valid location
                isValid = True
                for q_i in self._args:
                    if not q_i.locIteratorHasMatch():
                        isValid = False
                        break
                if not isValid:
                    break

                prev_pos = self._args[0].locIteratorGetMatch()
                isNear = True

                for i in range(1, len(self._args)):
                    q_i = self._args[i]

                    # Advance all other loc iterators to a position strictly after prev_pos
                    q_i.locIteratorAdvanceTo(prev_pos + 1)
                    if not q_i.locIteratorHasMatch():
                        isNear = False
                        break

                    cur_pos = q_i.locIteratorGetMatch()
                    if cur_pos - prev_pos > self.distance:
                        isNear = False
                        break

                    prev_pos = cur_pos

                if isNear:
                    # Collect positions
                    positions_list.append(prev_pos)
                
                    # Advance all loc iterators
                    for q_i in self._args:
                        q_i.locIteratorAdvance()
                else:
                    # Advance the first loc iterator
                    self._args[0].locIteratorAdvance()
                
            # If any matching positions were found, add them to the inverted list
            if len(positions_list) > 0:
                self.invertedList.appendPosting(maxDocid, positions_list)
            
            # Advance all doc iterators past maxDocid
            for q_i in self._args:
                q_i.docIteratorAdvancePast(maxDocid)
