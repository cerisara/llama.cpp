#!/bin/bash

s="Bonjour"

mod="/home/xtof/ggufs/qwen2.5-0.5b-instruct-q5_k_m.gguf"

source /home/xtof/envs/transformers/bin/activate
python xllamacpp.py --model "$mod" > ladder.log &
# python xllamacpp.py --model "$mod" --ladder mlp.pt > ladder.log &

# simule un Ctrl-C pour quitter proprement
PID=$!
cleanup() {
 kill -INT $PID 2>/dev/null || true   # SIGINT -> clean shutdown
 wait $PID 2>/dev/null || true       # now safe: the kill makes it exit
}
trap cleanup EXIT
# wait until the OpenAI endpoint is up
until curl -s http://127.0.0.1:8258/v1/models >/dev/null 2>&1; do
 sleep 1
done
echo "xllamacpp OAI endpoint found"

echo "$s" > tt
pi -e ./ladder-model.ts --provider ladder --model laddermodel --stream all < tt 2>&1 | tee hh.log  

