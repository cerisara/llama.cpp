source ~/envs/transformers/bin/activate

# before running this, you have to inspect which node is the unembedding node
embnode="result_output"
# TODO: modifier llama.cpp pour qu'il regarde tous les nodes de fin en cherchant
# le node qui contient une matrice qui ressemble à une matrice de unembedding

echo "save unembedding matrix"
echo 'La capitale de la Belgique est Bruxelles.' > tt
rm -f detembeds.*
NTOKS=0 SAVE_EMB="$embnode" python ./xllamacpp.py tt

NTOKS=1 python ./xllamacpp.py tt > repgld

echo 'La capitale de la Belgique est' > tt
NTOKS=5 python ./xllamacpp.py tt > repbad
mv tt_activs.npz actbad.npz
a=$(cat repbad | grep PROMPT_TOKENS | wc -w)
# contient le nb de tokens+1; l'index du token genere suivant = nb de tokens
# mais le cut suivant compte a partir de 1
 
goldtok=$(cat repgld | grep PROMPT_TOKENS | cut -c16- | cut -d',' -f$a)
echo "gold token $goldtok"
python ./xllamacpp.py tt $goldtok > repfix
cat repfix | grep GEN

