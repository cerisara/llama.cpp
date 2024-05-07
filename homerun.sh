txt="D'où l'eau de source Carola coule?"
./main -m /home/xtof/Downloads/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf -p '<|begin_of_text|><|start_header_id|>système<|end_header_id|>\n\nVous êtes un assistant intelligent.<|eot_id|>\n<|start_header_id|>utilisateur<|end_header_id|>\n\n'"$txt"'<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n' --temp 0

