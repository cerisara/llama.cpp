# test if it works with long input splitted into chunks

source ~/envs/transformers/bin/activate

cat q1_oracle.md > tt
echo "" >> tt
echo 'La capitale de la Belgique est Bruxelles.' >> tt

NTOKS=1 python ./xllamacpp.py --prompts tt > repgld
rm -f tt_activs.npz

cat q1_oracle.md > tt
echo "" >> tt
echo 'La capitale de la Belgique est' >> tt
NTOKS=5 python ./xllamacpp.py --prompts tt > repbad
mv tt_activs.npz actbad.npz
a=$(cat repbad | grep PROMPT_TOKENS | wc -w)
# contient le nb de tokens+1; l'index du token genere suivant = nb de tokens
# mais le cut suivant compte a partir de 1
 
goldtok=$(cat repgld | grep PROMPT_TOKENS | cut -c16- | cut -d',' -f$a)
echo "gold token $goldtok $a"
python ./xllamacpp.py --prompts tt --inject_token $goldtok > repfix
cat repfix | grep GEN

