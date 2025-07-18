# compile:

# make LLAMA_CUDA=1 main

# run:

p="Quel était le président de la république française en 1960 ?"

./main -ngl 1000 -m ~/nvme/llama3/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf -p '<|begin_of_text|><|start_header_id|>système<|end_header_id|>\n\nVous êtes un assistant intelligent.<|eot_id|>\n<|start_header_id|>utilisateur<|end_header_id|>\n\n'"$p"'<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n' -c 8192 --temp 0

