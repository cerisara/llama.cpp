source ~/envs/transformers/bin/activate

echo "save unembedding matrix"
echo 'La capitale de la Belgique est Bruxelles.' > tt
rm -f detembeds.*
NTOKS=0 SAVE_EMB=1 python ./xllamacpp.py tt > saveemb
rm tt_activs.npz

NTOKS=1 python ./xllamacpp.py tt > repgld
rm tt_activs.npz

echo 'La capitale de la Belgique est' > tt
NTOKS=5 python ./xllamacpp.py tt > repbad
mv tt_activs.npz actbad.npz
a=$(cat repbad | grep PROMPT_TOKENS | wc -w)
# contient le nb de tokens+1; l'index du token genere suivant = nb de tokens
# mais le cut suivant compte a partir de 1
 
goldtok=$(cat repgld | grep PROMPT_TOKENS | cut -c16- | cut -d',' -f$a)
echo "gold token $goldtok"
python ./xllamacpp.py tt $goldtok > repfix
rm tt_activs.npz
cat repfix | grep GEN

