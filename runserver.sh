#!/bin/bash

# WARNING: you must run first ./runfirst.sh
# The following test xllamacpp in OAI server mode through pi harness

s="Bonjour"

mod="/home/xtof/ggufs/Qwen3.5-9B-Q4_K_M.gguf"
mod="/home/xtof/ggufs/qwen2.5-0.5b-instruct-q5_k_m.gguf"

source /home/xtof/envs/transformers/bin/activate
rm -f cats.npz
TOKNOD=$(cat toknod.txt) python xllamacpp.py --model "$mod" --activs cats.npz > ladder.log &
# python xllamacpp.py --model "$mod" --ladder mlp.pt > ladder.log &

# wait until the OpenAI endpoint is up
until curl -s http://127.0.0.1:8258/v1/models >/dev/null 2>&1; do
 sleep 1
done
echo "xllamacpp OAI endpoint found"

echo "$s" > tt
pi -e ./ladder-model.ts --provider ladder --model laddermodel < tt 2>&1 | tee hh.log  

# print the real tokens consumed by the LLM: extract the token ids from
# the activation file, then detokenize them via the server /detokenize endpoint
# and split them into chatML chunks (one realtokens.chunks line per chunk)
TOKLIST=$(python read_activs.py cats.npz --tokens | grep '^TOKENS' | sed 's/^TOKENS //' | tr -s ' ')
echo "tokens consumed by the LLM:"
python realtokens.py "$TOKLIST"

echo "fini"
curl -X POST http://127.0.0.1:8258/shutdown

