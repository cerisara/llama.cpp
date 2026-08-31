source ~/envs/transformers/bin/activate

echo 'La capitale de la Belgique est Bruxelles.' > tt
python ./xllamacpp.py tt | grep PROMPT_TOKENS > ttt
echo 'La capitale de la Belgique est' > tt
python ./xllamacpp.py tt | grep PROMPT_TOKENS >> ttt
a=$(tail -1 ttt | wc -c)
goldtok=$(cut -c$a- ttt | cut -d',' -f1 | cut -c2- | head -1)
python ./xllamacpp.py tt $goldtok | grep GEN

