import sys
from RewriteWithPrf import RewriteWithPrf

class Rewriter:
    """
    The pipeline task that rewrites queries using Pseudo Relevance Feedback.
    """

    def __init__(self, parameters):
        """
        Initialize the Rewriter task.
        """
        self.params = parameters
        self.output_file_path = parameters.get('prf:expansionQueryFile')
        
        self.prf_model = RewriteWithPrf(parameters)

    def execute(self, batch):
        """
        Rewrite queries in the batch.

        batch format:
            qid: {
                'qstring': query_string,
                'ranking': [(score, externalId), ...]
            }

        Return:
            Updated batch with rewritten qstring.
        """
        output_file = None
        if self.output_file_path:
            try:
                output_file = open(self.output_file_path, 'w')
            except IOError as e:
                print(f"Error opening expansionQueryFile: {e}", file=sys.stderr)
        
        for qid, q_data in batch.items():
            if 'ranking' not in q_data or len(q_data['ranking']) == 0:
                continue

            current_ranking = q_data['ranking']
            original_query = q_data['qstring']

            # Perform PRF
            new_query, file_output_str = self.prf_model.rewrite(
                current_ranking, original_query
            )

            # Update Batch
            batch[qid]['qstring'] = new_query

            # Write to .qryOut file
            if output_file and file_output_str:
                output_file.write(f"{qid}: {file_output_str}\n")

        if output_file:
            output_file.close()

        return batch