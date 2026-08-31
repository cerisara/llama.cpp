source ~/envs/transformers/bin/activate

# before running this, you have to inspect which node is the unembedding node
embnode="embd"
# TODO: use LLM to guess the unembedding node name

echo "save unembedding matrix"
echo 'La capitale de la Belgique est Bruxelles.' > tt
rm -f detembeds.*
NTOKS=0 SAVE_EMB="$embnode" python ./xllamacpp.py tt
exit

python ./xllamacpp.py tt > repgld
cat repgld | grep PROMPT_TOKENS > ttt
echo 'La capitale de la Belgique est' > tt
python ./xllamacpp.py tt > repbad
cat repbad | grep PROMPT_TOKENS >> ttt
a=$(tail -1 ttt | wc -c)
goldtok=$(cut -c$a- ttt | cut -d',' -f1 | cut -c2- | head -1)
python ./xllamacpp.py tt $goldtok | grep GEN

