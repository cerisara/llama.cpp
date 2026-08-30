source ~/envs/transformers/bin/activate

echo 'La capitale de la Belgique est Bruxelles.' > tt
python ./xllamacpp.py tt > ttgoods.log
mv tt_activs.npz ttgoods.npz

echo 'La capitale de la Belgique est ' > tt
python ./xllamacpp.py tt > ttbads.log
mv tt_activs.npz ttbads.npz

python ./xllamacpp.py tt ttbads.npz > ttforce.log
grep GEN ttbads.log > /tmp/aa
grep GEN ttforce.log > /tmp/ab
if diff -q /tmp/aa /tmp/ab > /dev/null; then
    echo "BADSAME OK"
else
    echo "BADSAME KO"
	diff /tmp/aa /tmp/ab
fi

python ./xllamacpp.py tt ttgoods.npz > ttforce.log
grep GEN ttgoods.log > /tmp/aa
grep GEN ttforce.log > /tmp/ab
if diff -q /tmp/aa /tmp/ab > /dev/null; then
    echo "GOODSAME OK"
else
    echo "GOODSAME KO"
	diff /tmp/aa /tmp/ab
fi
 

