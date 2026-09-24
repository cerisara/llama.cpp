# source ~/envs/transformers/bin/activate
source ~/envs/hf/bin/activate

mod="/home/xtof/ggufs/qwen2.5-0.5b-instruct-q5_k_m.gguf"

ACTIVS=q1_oracle_activs.npz
rm -f "$ACTIVS"
LOGITS_ALL=1 NTOKS=1 python ./xllamacpp.py --model "$mod" --prompts ~/git/researchplm/agentic/q1_oracle.md

# train on the single activations file (tokens are stored inside it)
python train.py "$ACTIVS"

# TODO: on-policy distillation: oracle prompt + rollout + forward to get acts + train.py
