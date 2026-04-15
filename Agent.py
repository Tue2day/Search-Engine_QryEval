"""
Pipeline agents.
"""

# Copyright (c) 2026, Carnegie Mellon University.  All Rights Reserved.

import json
import os

from RagSupport import (
    DEFAULT_DENSE_MODEL_PATH,
    DEFAULT_LLM_ADDRESS,
    build_rag_prompt,
    normalize_llm_response,
    select_rag_passage,
    send_to_llm,
    write_prompt_file)


class Agent:
    """
    Consume search results and produce downstream agent outputs.
    """

    def __init__(self, parameters):
        agent_type = parameters.get('type', '').lower()
        if agent_type != 'rag':
            raise Exception(f'Error: Unknown agent type {parameters.get("type")}.')

        self._agent = RagAgent(parameters)


    def execute(self, batch):
        return(self._agent.execute(batch))


class RagAgent:
    """
    Retrieval-augmented generation over ranked documents.
    """

    def __init__(self, parameters):
        self._parameters = parameters
        self._agent_depth = int(parameters.get('agentDepth', 1))
        self._model_server = parameters.get(
            'rag:modelServer', DEFAULT_LLM_ADDRESS)
        self._dense_model_path = parameters.get(
            'rag:dense:modelPath',
            parameters.get('dense:modelPath', DEFAULT_DENSE_MODEL_PATH))
        self._prompt_path = parameters.get('rag:promptPath')
        self._psg_len = int(parameters.get('rag:psgLen', 0))
        self._psg_stride = int(parameters.get('rag:psgStride', 0))
        self._psg_cnt = int(parameters.get('rag:psgCnt', 1))
        self._max_title_length = int(parameters.get('rag:maxTitleLength', 0))
        self._prompt_id = int(parameters.get('rag:prompt', 1))
        self._authorize_message = self._build_authorize_message(parameters)

        if self._agent_depth <= 0:
            raise Exception('Error: agentDepth must be > 0.')
    def execute(self, batch):
        prompts_by_qid = {}

        for qid in batch:
            qstring = batch[qid]['qstring']
            ranking = batch[qid].get('ranking', [])
            top_ranking = ranking[:self._agent_depth]

            passages = []
            for _, external_id in top_ranking:
                passages.append(select_rag_passage(
                    qstring,
                    external_id,
                    self._dense_model_path,
                    self._psg_len,
                    self._psg_stride,
                    self._psg_cnt,
                    self._max_title_length))

            prompt = build_rag_prompt(
                self._prompt_id,
                qstring,
                passages,
                dict(self._authorize_message))
            prompts_by_qid[qid] = prompt
            batch[qid]['ragPrompt'] = prompt
            batch[qid]['ragPassages'] = passages

            answer = send_to_llm(self._model_server, prompt)
            batch[qid]['answer'] = normalize_llm_response(answer)

        if self._prompt_path is not None:
            write_prompt_file(self._prompt_path, prompts_by_qid)

        return(batch)


    def _build_authorize_message(self, parameters):
        email = None
        code = None

        email = os.environ.get('CMU_ANDREW_EMAIL')
        code = os.environ.get('CMU_LLM_ACCESS_CODE')

        if email is None or code is None:
            auth = self._read_auth_file(parameters.get('rag:authPath'))
            if email is None:
                email = auth.get('email')
            if code is None:
                code = auth.get('code')

        if email is None:
            email = 'unknown@andrew.cmu.edu'
        if code is None:
            code = 'unknown'

        return({
            'role': 'authorize',
            'email': email,
            'code': code
        })


    def _read_auth_file(self, auth_path):
        candidate_paths = []

        if auth_path is not None:
            candidate_paths.append(auth_path)

        candidate_paths.extend([
            'llm_auth.json',
            os.path.join('INPUT_DIR', 'llm_auth.json')
        ])

        for path in candidate_paths:
            if path is None or not os.path.exists(path):
                continue

            try:
                with open(path, 'r', encoding='utf-8') as handle:
                    data = json.load(handle)
                if isinstance(data, dict):
                    return(data)
            except Exception:
                continue

        return({})
