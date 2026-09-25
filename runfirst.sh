# must be run FIRST before every session
# sets-up every file required and runs a safe check
# also, don't forget to cmake llama-server first!

source ~/envs/transformers/bin/activate

mod="/home/xtof/ggufs/Qwen3.5-9B-Q4_K_M.gguf"
mod="/home/xtof/ggufs/qwen2.5-0.5b-instruct-q5_k_m.gguf"

echo "save unembedding matrix"
echo 'La capitale de la Belgique est Bruxelles.' > tt
rm -f detembeds.*
rm -f tt_activs.npz
NTOKS=1 SAVE_EMB=1 python ./xllamacpp.py --model "$mod" --prompts tt > saveemb
python ./init_layers.py saveemb

rm -f tt_activs.npz
TOKNOD=$(cat toknod.txt) NTOKS=1 python ./xllamacpp.py --model "$mod" --prompts tt > repgld
python read_activs.py tt_activs.npz --tokens | grep -a TOKENS > toksgld
mv tt_activs.npz actgld.npz

echo 'La capitale de la Belgique est' > tt
TOKNOD=$(cat toknod.txt) NTOKS=5 python ./xllamacpp.py --model "$mod" --prompts tt > repbad
ntoks=$(python read_activs.py tt_activs.npz --tokens | grep -a TOKENS | wc -w)
ntoksq=$((ntoks - 3))
echo "ntoks question $ntoksq"
mv tt_activs.npz actbad.npz
goldtok=$(cat toksgld | cut -d' ' -f$ntoksq)
echo "gold token $goldtok"
python ./xllamacpp.py --model "$mod" --prompts tt --inject_token $goldtok > repfix
rm -f tt_activs.npz
cat repfix | grep -a GEN

# the next test you may want to run is ./runserver.sh

