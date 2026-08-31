source ~/envs/transformers/bin/activate

# plus simple que unittest1: ne passe pas par la matrice de unembeddings,
# mais sauve les embeddings du gold avec /embeddings pour recuperer l'embedding
# de "Bruxelles", puis injecte cet embed
#
# BUG: ca ne peut pas marcher, car la phrase "gold" sort un vecteur de
# prediction, qui est different du vecteur gold des embeddings

echo 'La capitale de la Belgique est Bruxelles.' > tt
NTOKS=0 python ./xllamacpp.py tt > repgld
mv tt_activs.npz actgld.npz

echo 'La capitale de la Belgique est' > tt
NTOKS=5 python ./xllamacpp.py tt > repbad
mv tt_activs.npz actbad.npz
a=$(cat repbad | grep PROMPT_TOKENS | wc -w)
# contient le nb de tokens+1; on veut l'index du token genere suivant = nb de tokens
a=$((a - 1))

echo 'target activ'
python read_activs.py actgld.npz 1 $a
# above cmd also savec onevec.bin

python ./xllamacpp.py tt onevec.bin > repfix
cat repfix | grep GEN

