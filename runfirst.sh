# must be run FIRST before every session
# sets-up every file required and runs a safe check
# also, don't forget to cmake llama-server first!

# source ~/envs/transformers/bin/activate

mod="/home/xtof/ggufs/qwen2.5-0.5b-instruct-q5_k_m.gguf"

echo "save unembedding matrix"
echo 'La capitale de la Belgique est Bruxelles.' > tt
rm -f detembeds.*
NTOKS=1 SAVE_EMB=1 python ./xllamacpp.py --model "$mod" --prompts tt > saveemb
rm -f tt_activs.npz
python ./init_layers.py saveemb

TOKNOD=$(cat toknod.txt) NTOKS=1 python ./xllamacpp.py --model "$mod" --prompts tt > repgld
rm -f tt_activs.npz

echo 'La capitale de la Belgique est' > tt
NTOKS=5 python ./xllamacpp.py --model "$mod" --prompts tt > repbad
mv tt_activs.npz actbad.npz
a=$(cat repbad | grep PROMPT_TOKENS | wc -w)
# contient le nb de tokens+1; l'index du token genere suivant = nb de tokens
# mais le cut suivant compte a partir de 1
 
goldtok=$(cat repgld | grep PROMPT_TOKENS | cut -c16- | cut -d',' -f$a)
echo "gold token $goldtok"
python ./xllamacpp.py --model "$mod" --prompts tt --inject_token $goldtok > repfix
rm -f tt_activs.npz
cat repfix | grep GEN

