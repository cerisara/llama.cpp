# make GGML_CUDA=1 llama-eval-callback
#
# see https://qwen.readthedocs.io/en/v1.5/run_locally/llama.cpp.html

# python getdata.py > frinstr.txt

modnom="/home/xtof/qwen2.5-0.5b-instruct-q5_k_m.gguf"

rm /dev/shm/sem.py2c_sem
rm /dev/shm/sem.c2py_sem
rm -f layers2save
touch layers2save
echo 'l_out-10' >> layers2save
echo 'l_out-12' >> layers2save
echo 'result_norm' >> layers2save

. /home/envs/hf/bin/activate
echo "notrain" > pyargs.txt
python pythonlisten.py > py.log &

echo "What comes after 5? Answer: " > toto.txt
tf="toto.txt"

while IFS="" read -r p || [ -n "$p" ]
do
    rm -rf detlog
    mkdir detlog
    ./llama-cli --logdir detlog --temp 0.7 -c 2048 -nkvo -m "$modnom" -p "$p" -n 20 # -fa -ngl 100
done < "$tf"

echo "quit" > pyargs.txt

