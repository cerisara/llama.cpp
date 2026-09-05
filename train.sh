# source ~/envs/transformers/bin/activate
source ~/envs/hf/bin/activate

mod="/home/xtof/ggufs/qwen2.5-0.5b-instruct-q5_k_m.gguf"

rm q1_oracle_activs.npz
LOGITS_ALL=1 NTOKS=1 python ./xllamacpp.py --model "$mod" --prompts ~/git/researchplm/agentic/q1_oracle.md > q1.log


