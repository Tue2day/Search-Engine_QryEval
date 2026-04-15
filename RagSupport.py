"""
Shared utilities for dense retrieval, prompting, and answer outputs.
"""

# Copyright (c) 2026, Carnegie Mellon University.  All Rights Reserved.

import json
import os
import socket

import numpy as np

from Idx import Idx
from DenseEncoder import DenseEncoder
from PassageGenerator import PassageGenerator

DEFAULT_DENSE_INDEX_PATH = 'INPUT_DIR/index-cw22b-wp-faiss-b300-Fp'
DEFAULT_DENSE_MODEL_PATH = 'INPUT_DIR/co-condenser-marco-retriever'
DEFAULT_LLM_ADDRESS = '128.2.204.71/59596'


def ensure_parent_dir(path):
    if path is None:
        return
    parent = os.path.dirname(path)
    if parent != '':
        os.makedirs(parent, exist_ok=True)


def get_dense_encoder(model_path):
    if model_path is None:
        model_path = DEFAULT_DENSE_MODEL_PATH
    return(DenseEncoder(model_path))


def load_faiss_index(index_path):
    try:
        import faiss
    except ImportError as e:
        raise Exception(
            'Error: Dense retrieval requires faiss to be installed in the '
            'active Python environment.') from e

    if index_path is None:
        index_path = DEFAULT_DENSE_INDEX_PATH

    return(faiss.read_index(index_path))


def get_document_strings(external_id):
    internal_id = Idx.getInternalDocid(external_id)
    if internal_id is None or internal_id < 0:
        raise Exception(f'Error: Unknown document id {external_id}.')

    title_string = Idx.getAttribute('title-string', internal_id) or ''
    body_string = Idx.getAttribute('body-string', internal_id) or ''
    return(title_string, body_string)


def select_rag_passage(qstring, external_id, dense_model_path,
                       psg_len, psg_stride, psg_cnt, max_title_length):
    title_string, body_string = get_document_strings(external_id)
    passage_generator = PassageGenerator(
        psg_len, psg_stride, psg_cnt, max_title_length)
    passages = passage_generator.build_passages(title_string, body_string)

    if len(passages) == 1 or int(psg_cnt) <= 1:
        return(passages[0])

    encoder = get_dense_encoder(dense_model_path)
    query_vector = encoder.encode(qstring)
    passage_vectors = encoder.encode(passages)
    scores = np.dot(passage_vectors, query_vector)
    best_index = int(np.argmax(scores))
    return(passages[best_index])


def build_rag_prompt(prompt_id, qstring, passages, authorize_message):
    prompt_id = int(prompt_id)

    prompt_builders = {
        1: lambda: (
            'You are a helpful chatbot. Use the passage below to provide '
            "a short answer to the specified question. Don't explain "
            'your reasoning. Just generate a short answer.',
            f'Question: {qstring}\nContext:{compact_context}\n'
            f'Question: {qstring}\nAnswer: '
        ),
        2: lambda: (
            'You are a question answering system. Answer the question '
            'using only the information provided in the context. '
            'Provide a short answer only.',
            f'Question: {qstring}\nContext: {compact_context}\nAnswer:'
        ),
        3: lambda: (
            'Answer the question with a very short phrase (one or two '
            'words if possible). Do not include explanations.',
            f'Question: {qstring}\nContext: {compact_context}\nAnswer:'
        ),
        4: lambda: (
            'The context may contain irrelevant or misleading information. '
            'Identify the most relevant part and use it to answer the '
            'question. Provide a short answer without explanation.',
            f'Question: {qstring}\nContext: {compact_context}\nAnswer:'
        ),
        5: lambda: (
            'Find the exact answer from the context and copy it directly. '
            'Do not rephrase or explain.',
            f'Question: {qstring}\nContext: {compact_context}\nAnswer:'
        ),
        6: lambda: (
            'First think about the answer based on the context, then '
            'provide the final answer in a short phrase. Only output the '
            'final answer.',
            f'Question: {qstring}\nContext: {compact_context}\nAnswer:'
        ),
        7: lambda: (
            'Use retrieval evidence to answer briefly. If multiple passages '
            'repeat the same fact, trust that fact.',
            f'Question: {qstring}\nEvidence snippets:\n{context}\n'
            'Short answer:'
        ),
        8: lambda: (
            'Identify the answer that best matches the question from the '
            'retrieved passages. Output only the answer.',
            f'Question: {qstring}\nRetrieved passages:\n{context}\nAnswer:'
        ),
        9: lambda: (
            'Respond with a minimal factoid answer supported by the passages.',
            f'Question: {qstring}\nSupport:\n{context}\nAnswer:'
        ),
        10: lambda: (
            'You answer trivia questions with short exact strings. '
            'Use the passages and avoid extra words.',
            f'Question: {qstring}\nPassages:\n{context}\nExact answer:'
        )
    }

    if prompt_id not in prompt_builders:
        raise Exception(f'Error: Unknown rag:prompt value {prompt_id}.')

    working_passages = list(passages)
    while True:
        context = '\n\n'.join(
            f'Passage {i + 1}: {passage}'
            for i, passage in enumerate(working_passages))
        compact_context = ' '.join(working_passages)

        system_text, user_text = prompt_builders[prompt_id]()
        prompt = [{'role': 'authorize', 'email': "yuzhong4@andrew.cmu.edu",  'code': 'AX0a'},
                  {'role': 'system', 'content': system_text},
                  {'role': 'user', 'content': user_text}]

        if _prompt_size_bytes(prompt) <= (20 * 1024) - 64:
            return(prompt)
        if len(working_passages) <= 1:
            return(_truncate_prompt_text(prompt))

        working_passages = working_passages[:-1]


def normalize_llm_response(response):
    if response is None:
        return('')

    response = str(response).strip()
    if response == '':
        return('')

    if response.startswith('EXCEPTION::'):
        return(response)

    lines = [line.strip() for line in response.splitlines() if line.strip() != '']
    if len(lines) == 0:
        return('')

    response = lines[0]
    if response.lower() in ['answer:', 'final answer:', 'short answer:', 'exact answer:'] and len(lines) > 1:
        response = lines[1]

    lower = response.lower()
    prefixes = ['answer:', 'short answer:', 'final answer:', 'exact answer:']
    for prefix in prefixes:
        if lower.startswith(prefix):
            response = response[len(prefix):].strip()
            break

    if len(response) >= 2 and response[0] == response[-1] and response[0] in ['"', "'"]:
        response = response[1:-1].strip()

    return(response)


def send_to_llm(llm_address, messages):
    """
    Send a list of messages to an LLM server and get a response.
    """
    max_msg_bytes = 20 * 1024

    try:
        message = json.dumps(messages)
        message = message.encode() + bytearray(b'\0\0\0\0')

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            llm_host, llm_port = llm_address.split('/')
            sock.connect((llm_host, int(llm_port)))
            sock.sendall(message)

            data = bytearray()
            while len(data) < max_msg_bytes:
                packet = sock.recv(max_msg_bytes - len(data))
                if not packet:
                    break
                data.extend(packet)
                if data[-4:] == bytearray(b'\0\0\0\0'):
                    data = data[0:-4]
                    break

            response = data.decode()
    except Exception as e:
        response = f'EXCEPTION:: {str(e)}'

    response = response.strip()
    response = ' '.join(response.split())

    return(normalize_llm_response(response))


def write_prompt_file(path, prompts_by_qid):
    ensure_parent_dir(path)
    lines = []
    for qid, prompt in prompts_by_qid.items():
        lines.append(f'{qid}: {json.dumps(prompt)}')
    with open(path, 'w', encoding='utf-8') as handle:
        for line in lines:
            handle.write(line + '\n')


def write_qain_file(path, answers_by_qid):
    ensure_parent_dir(path)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(answers_by_qid, handle, ensure_ascii=False)


def _prompt_size_bytes(prompt):
    return(len(json.dumps(prompt).encode('utf-8')))


def _truncate_prompt_text(prompt):
    max_payload_bytes = (20 * 1024) - 64
    trimmed_prompt = list(prompt)

    while _prompt_size_bytes(trimmed_prompt) > max_payload_bytes:
        user_content = trimmed_prompt[-1]['content']
        if len(user_content) <= 256:
            break
        trimmed_prompt[-1]['content'] = user_content[:-256]

    return(trimmed_prompt)
