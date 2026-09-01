source ~/envs/transformers/bin/activate

rm q1_oracle_activs.npz
NTOKS=1 python ./xllamacpp.py ~/git/researchplm/agentic/q1_oracle.md > q1.log


