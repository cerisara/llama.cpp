# Using this patched llama.cpp — user guide

This checkout is **not vanilla llama.cpp**. It is patched for a research project
on "ladders": small side-networks trained on top of a *frozen* quantized LLM.
The patch does two things:

1. it can **show what is happening inside** the model (dump the internal
   activations of a given layer to a file), and
2. it can **nudge the model** at inference time by adding a small learned bias
   to the final embedding before the output layer (the "ladder").

Everything else behaves like normal llama.cpp. All the technical details
(env vars, file formats, internals, performance, gotchas) are in **AGENTS.md**;
this page is just how to use it.

## The three modes (four ways to run it)

| what you want                                     | mode                      | python needed?       |
| ------------------------------------------------- | ------------------------- | -------------------- |
| normal fast server (benchmark, eval, sampling)    | **A — plain**             | no                   |
| same, with a trained ladder folded into the graph | **A — graph ladder**      | no                   |
| collect activations to train a ladder             | **B — capture**           | yes (`xllamacpp.py`) |
| run a ladder without the python round-trip        | **C — native C++ ladder** | no                   |

The build is the same for all of them; the mode is chosen by environment
variables and by a small file named `layers2save` in the directory you launch
from.

### 1. Plain server

```bash
cd /home/xtof/git/llama.cpp
: > layers2save
DETSON_OFF=1 ./build/bin/llama-server -m MODEL.gguf -ngl 99 -c 8192 \
    --port 8082 --no-webui --no-warmup
```

`DETSON_OFF=1` is required: without it the activation hook is always installed
and everything becomes ~6x slower. (`build/bin` and `build_g/bin` are the same
configuration; the experiments use `build_g` for the graph ladder below.)

### 2. Plain server + ladder folded into the model graph (recommended)

```bash
# turn a trained mlp.pt into the graph format
~/envs/transformers/bin/python3 ../researchplm/ladder/cppladder_export_graph.py \
    --mlp logdir/xp_xxx_mlp.pt --out logdir/xp_xxx_graph.bin

DETSON_OFF=1 LADDER_BIN=/abs/path/xp_xxx_graph.bin LADDER_LAYER=19 \
    ./build_g/bin/llama-server -m MODEL.gguf -ngl 99 -c 8192 \
    --port 8082 --no-webui --no-warmup
```

Fast, no python, no shared memory. Works for the `nemotron_h` (Luciole) and
`qwen35` architectures only; other models silently ignore `LADDER_BIN`.

### 3. Capture activations (needs the python proxy)

```bash
cd /home/xtof/git/llama.cpp
printf 'l_out-19\nresult_norm\n' > layers2save      # one newline per line!

LOGITS_ALL=1 ~/envs/transformers/bin/python3 xllamacpp.py \
    --model MODEL.gguf --port 8261 --activs /abs/path/activs.bin
```

`xllamacpp.py` starts the server, drains the shared memory, and writes every
hooked tensor to `--activs` (gzip chunks). The server **blocks** for each hooked
layer until python answers, so it must always be started through
`xllamacpp.py` — a server with a non-empty `layers2save` and no python client
hangs forever.

### 4. Native C++ ladder (no python)

```bash
printf 'l_out-19\nresult_norm\n' > layers2save
~/envs/transformers/bin/python3 ../researchplm/ladder/cppladder_export.py \
    --mlp logdir/xp_xxx_mlp.pt --out logdir/xp_xxx_cpp.bin

LADDER_CPP=/abs/path/xp_xxx_cpp.bin \
    ~/envs/transformers/bin/python3 xllamacpp.py --model MODEL.gguf --port 8261
```

The ladder is computed inside llama.cpp; the weights file is hot-reloaded when
it changes on disk (handy when the policy is being updated between rollouts).

## How to tell it worked

* plain: `DETSON_OFF set: detson callback removed (plain mode)`
* graph ladder: `ladder: ... (8192x4096 f16) on CUDA0, injected at l_out-19`
* capture: `capture server ready` / rows written to the `--activs` file
* native ladder: `cppladder: native ladder installed from ... (nodes: ...)`

## Four things that bite

1. **Never rebuild a build directory while a server from it is running** — the
   linker rewrites the `.so` in place and corrupts the live process. Use the
   other build dir (`build/` vs `build_g/`) or stop the server first.
2. `layers2save` is read from the **current directory**, not from the build
   dir. An empty/absent file + `DETSON_OFF=1` = plain mode; a non-empty file
   without the python client = deadlock.
3. Both builds (`build/`, `build_g/`) are the same configuration; `xllamacpp.py`
   always uses `build/`, the experiments' graph server uses `build_g/`.
4. Only `nemotron_h` and `qwen35` have the in-graph ladder hook.

## Where to learn more

* **AGENTS.md** (this directory) — the technical reference: the three modes,
  env vars, file formats, internals of the shared-memory handshake, performance
  numbers, build rules, known issues.
* `../researchplm/ladder/cppladder_README.md` — the native ladder and graph
  mode, from the research repo.
* `README.md` — upstream llama.cpp documentation.
