#!/bin/bash
source ~/envs/transformers/bin/activate
# source ~/envs/hf/bin/activate

mod="/home/xtof/ggufs/qwen2.5-0.5b-instruct-q5_k_m.gguf"
PROMPTS=~/git/researchplm/agentic/q1_oracle.md

ACTIVS=q1_oracle_activs.npz
LOG="server.log"
HOST=127.0.0.1
PORT=8258

rm -f "$ACTIVS" "$LOG"

# launch xllamacpp as a local OpenAI proxy server; activations are streamed to
# $ACTIVS on disk as tokens are generated (--activs keeps the server running).
# PYTHONUNBUFFERED=1 is required: stdout is redirected to a file, and without it
# Python block-buffers, so the readiness banner below would never be flushed.
LOGITS_ALL=1 NTOKS=1 PYTHONUNBUFFERED=1 \
    nohup python ./xllamacpp.py --model "$mod" \
        --host "$HOST" --port "$PORT" --activs "$ACTIVS" > "$LOG" 2>&1 &
SRV=$!

# wait until the OpenAI endpoint is ready
for i in $(seq 1 120); do
    grep -q "OpenAI endpoint on http://$HOST:$PORT/v1" "$LOG" 2>/dev/null && break
    kill -0 $SRV 2>/dev/null || { echo "xllamacpp server died:"; tail -50 "$LOG"; exit 1; }
    sleep 1
done
grep -q "OpenAI endpoint on http://$HOST:$PORT/v1" "$LOG" || { echo "server did not become ready:"; tail -50 "$LOG"; kill $SRV; exit 1; }
echo "server ready"

# build the chat payload from the prompt file and send it via curl
python - "$PROMPTS" > payload.json <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    user = f.read()
print(json.dumps({
    "model": "ignored",
    "messages": [{"role": "user", "content": user}],
    "stream": False,
}))
PY
curl -s -X POST "http://$HOST:$PORT/v1/chat/completions" \
     -H "Content-Type: application/json" -d @payload.json
echo

# stop the proxy server (closes and finalizes the activations file)
curl -s -X POST "http://$HOST:$PORT/shutdown" > /dev/null
wait $SRV

exit
# train on the single activations file (tokens are stored inside it)
python train.py "$ACTIVS"

# TODO: on-policy distillation: oracle prompt + rollout + forward to get acts + train.py
