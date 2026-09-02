# source ~/envs/transformers/bin/activate
source ~/envs/hf/bin/activate

rm q1_oracle_activs.npz
LOGITS_ALL=1 NTOKS=1 python ./xllamacpp.py ~/git/researchplm/agentic/q1_oracle.md > q1.log


