#!/usr/bin/env python3
"""Analyse the tokens consumed by the LLM during a runserver.sh session.

Reads the space-separated token ids given as the first argument and performs
two successive tasks:

  1. Detokenize the whole sequence via llama-server's /detokenize endpoint,
     write the decoded text to realtokens.txt and print it as REAL_TOKENS.

  2. Split the sequence into chatML chunks (each delimited by
     <|im_start|><role>...<|im_end|>...; role is system / user / tool call /
     tool result / assistant / ...) and write, to realtokens.chunks, one line
     per chunk holding the token index of its <|im_start|> and its role.
"""
import json
import sys
import urllib.request

DETOKENIZE_URL = "http://127.0.0.1:8257/detokenize"
REALTOKENS_FILE = "realtokens.txt"
CHUNKS_FILE = "realtokens.chunks"

IM_START = 151644  # <|im_start|>
IM_END = 151645    # <|im_end|>
IM_START_TEXT = "<|im_start|>"


def detokenize(tokens):
    """Return the decoded text for the given list of token ids."""
    req = urllib.request.Request(
        DETOKENIZE_URL,
        data=json.dumps({"tokens": [int(t) for t in tokens]}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as r:
        return json.load(r)["content"]


def save_realtokens_text(tokens):
    """Task 1: detokenize everything and write it to realtokens.txt."""
    text = detokenize(tokens)
    print("REAL_TOKENS", text)
    with open(REALTOKENS_FILE, "w") as f:
        f.write(text)
    return text


def save_realtokens_chunks(tokens):
    """Task 2: split the tokens into chatML chunks and write realtokens.chunks.

    Returns the list of (start, end, role) triplets for each chunk, where
    start/end are the inclusive token indices of the chunk's <|im_start|> and
    matching <|im_end|> (end == len(tokens)-1 when no <|im_end|> yet).
    """
    # each chunk begins with an <|im_start|> and ends at the matching
    # <|im_end|>; the last assistant chunk may have no <|im_end|> yet because
    # generation is still ongoing, so it runs to the end of the sequence.
    starts = [i for i, t in enumerate(tokens) if t == IM_START]
    ends = [i for i, t in enumerate(tokens) if t == IM_END]

    chunks = []
    for s in starts:
        # chunk spans [s, e] inclusive where e is the next im_end after s
        e = next((x for x in ends if x > s), len(tokens) - 1)
        # role = first line after <|im_start|> in the decoded chunk text
        role = ""
        text = detokenize(tokens[s : e + 1])
        i = text.find(IM_START_TEXT)
        if i >= 0:
            role = text[i + len(IM_START_TEXT):].split("\n", 1)[0].strip()
        chunks.append((s, e, role))

    with open(CHUNKS_FILE, "w") as f:
        for s, e, role in chunks:
            f.write(f"{s}\t{role}\n")
    print("CHUNKS")
    for s, e, role in chunks:
        print("CHUNK", s, role)

    # sanity check: assume the last chunk is the (still-generating) assistant
    # reply; extract its token ids, detokenize them and print them as a check
    if chunks:
        s, e, role = chunks[-1]
        last_toks = tokens[s : e + 1]
        print("LAST_ASSISTANT_TOKENS", " ".join(str(t) for t in last_toks))
        print("LAST_ASSISTANT_TEXT", detokenize(last_toks))
    return chunks


def main():
    if len(sys.argv) < 2:
        print("usage: realtokens.py '<space separated token ids>'", file=sys.stderr)
        sys.exit(1)
    tokens = [int(t) for t in sys.argv[1].split()]
    print("DETOKEN", tokens)
    save_realtokens_text(tokens)
    save_realtokens_chunks(tokens)


if __name__ == "__main__":
    main()
