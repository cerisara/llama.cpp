# cmake -B build
# cmake --build built --config release

p="/home/xtof/git/researchplm/agentic/q1_oracle.md"

# ./build/bin/llama-perplexity -m ~/ggufs/qwen2.5-0.5b-f32.gguf --file "$p" -ngl 99

./build/bin/llama-finetune -m ~/ggufs/qwen2.5-0.5b-f32.gguf --file "$p" -t 1 -ngl 99 -epochs 1 -c 1024 -o toto.gguf
exit

./build/bin/llama-perplexity -m toto.gguf --file "$p" -ngl 99

