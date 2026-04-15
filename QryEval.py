# Copyright (c) 2026, Carnegie Mellon University.  All Rights Reserved.

import cProfile
import json
import os
import sys

import Util

from Idx import Idx
from Agent import Agent
from Output import Output
from Ranker import Ranker
from Reranker import Reranker
from Rewriter import Rewriter
from TeIn import TeIn
from Timer import Timer

# ------------------ Global variables ---------------------- #

usage = "Usage:  python QryEval.py paramFile\n\n"


# ------------------ Methods (alphabetical) ---------------- #

def main ():
    """The main function"""

    timer = Timer()
    timer.start()

    # Initialize the index and experiment parameters.
    parameters = readParameterFile()
    parameters = propagate_shared_parameters(parameters)
    Idx.open(parameters['indexPath'])
    queries = Util.read_queries(parameters['queryFilePath'])

    # Run the ranking pipeline.  In most search engines, the pipeline
    # evaluates 1 query at a time. Some of the tools used in HW2-HW5
    # are faster if called 1 time * n queries instead of n times * 1
    # query, so each stage of our pipeline does all queries and then
    # passes its results to the next stage.
    # 
    # Start by creating an initial batch object. Tasks will update it
    # as they produce new information.
    batch = {qid: {'qstring': qstring} for qid, qstring in queries.items()}
    task_list = [t for t in parameters.keys() if t.startswith('task_')]


    for task_name in task_list:
        print(f'\n-- {task_name}: {parameters[task_name]["type"]} --\n')
        task_type = task_name.split(':')[1]
        task_parameters = parameters[task_name]

        if task_type == 'agent':
            task = Agent(task_parameters)
        elif task_type == 'output':
            task = Output(task_parameters)
        elif task_type == 'ranker':
            task = Ranker(task_parameters)
        elif task_type == 'rewriter':
            task = Rewriter(task_parameters)
        elif task_type == 'reranker':
            task = Reranker(task_parameters)
        else:
            print(f'Error: Unexpected key {k} in {param_path}')

        batch = task.execute(batch)
    
    # Clean up
    Idx.close()
    timer.stop()
    print('Time:  ' + str(timer))


def readParameterFile():
    """
    Get a dict that contains the contents of the parameter file.
    """

    # Remind the forgetful.
    if len(sys.argv) != 2:
        print(usage)
        sys.exit(1)

    if not os.path.exists(sys.argv[1]):
        print('No such file {}.'.format(sys.argv[1]))
        sys.exit(1)

    # Read the .param file into a dict. Values that look like numbers
    # are stored as numbers.
    with open(sys.argv[1]) as f:
        d = json.load(f)
    d = Util.str_to_num(d)

    return(d)


def propagate_shared_parameters(parameters):
    """
    Copy shared dense settings from the dense ranker into later tasks
    when those tasks omit them. This matches the HW5 guide's expectation
    that DPR and RAG often share one encoder model.
    """
    dense_model_path = None
    for task_name, task_params in parameters.items():
        if (task_name.startswith('task_') and
                task_name.endswith(':ranker') and
                task_params.get('type') == 'dense'):
            dense_model_path = task_params.get('dense:modelPath')
            break

    if dense_model_path is None:
        return(parameters)

    for task_name, task_params in parameters.items():
        if task_name.startswith('task_') and task_name.endswith(':agent'):
            task_params.setdefault('rag:dense:modelPath', dense_model_path)

    return(parameters)


# ------------------ Script body --------------------------- #

if __name__ == '__main__':
    main()
