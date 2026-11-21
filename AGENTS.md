# AGENTS.md — xtof's patched llama.cpp (research fork)

Read this before touching anything here. This is **not** a vanilla llama.cpp
checkout: it carries a working-tree patch used by the "ladder" side-network
research (test-time adaptation / rejection-sampling post-training of a frozen
quantized LLM), and it is normally **in use by live experiments**.

## 1. What this fork is for

The research (see `../researchplm/ladder/` and `../researchplm/luciole/`)
freezes a quantized backbone (served by `llama-server`) and adds a small dense
"ladder" MLP applied **after the last block, before the LM head**:

```
x   = concat(l_out-K , result_norm)          # K = LADDER_LAYER, default 19
b   = vals @ (keys . x / ||x||)              # dense cosine gate (no relu/thr)
y   = result_norm + b                        # y feeds output.weight (the logits)
```

To train that ladder (and the reward head on top of it), the experiments need
two extra capabilities that stock llama.cpp does not have:

1. **Dump intermediate activations** (`l_out-19`, `result_norm`) out of a
   forward pass — for reward features and for the SFT targets.
2. **Inject the ladder at inference time**, fast enough to be used in a
   rollout loop (generation + eval).

Hence the patch. Consumers (do not break their expectations):

| consumer                                | file                              | what it needs                    |
| --------------------------------------- | --------------------------------- | -------------------------------- |
| `luciole/b0.py run`                     | `~/git/researchplm/luciole/b0.py` | mode A + mode B                  |
| `luciole/xp_sqa*_base.py`               | same dir                          | plain server (mode A, no ladder) |
| `ladder/fqbench.py`, `ladder/train*.py` | `~/git/researchplm/ladder/`       | mode B                           |
| `ladder/cppladder_*`                    | same dir                          | mode C                           |

### Repository state (important)

* branch: `sharedram2026` (base: `ggml-org/llama.cpp`).
* remotes: `origin` = upstream ggml-org, `a` = `github:cerisara/llama.cpp`.
* **The patch is committed**: `15c53ffad` (llama.cpp + `xllamacpp.py`), the
  `xllamacpp.py` robustness fixes came in through the merge `c1f1b1cff`, and
  `430d63bae` added `AGENTS.md` + `README.xtof.md`. So a fresh clone of HEAD
  already contains the three modes and both docs.
  **Still untracked**: `tools/server/cppladder_ladder.h` and
  `tools/server/cppladder_server.cpp` (mode C needs the header!) — a fresh
  clone of HEAD **cannot build mode C**. Copy them from
  `../researchplm/ladder/` (or from this working tree) first.
  Do not run `cppladder_deploy.sh` blindly either: it does
  `git checkout HEAD -- tools/server/server.cpp xllamacpp.py` (safe now) but it
  also regenerates those two files from `../researchplm/ladder/` — check the
  diff before rebuilding.
* `tools/server/cppladder_server.cpp` is currently a **byte-identical copy of
  `tools/server/server.cpp`** (a build artefact of `cppladder_gen.py`); the
  canonical sources live in `../researchplm/ladder/` (`cppladder_*.py`,
  `cppladder_ladder.h`, `cppladder_server.cpp`).
* `README.xtof.md` is a short **layman/user guide** (no internals: they live
  here and in `../researchplm/ladder/cppladder_README.md`). The shared-memory
  handshake internals are documented in §7 below.

## 2. The three runtime modes

The **build is the same for all three** (`build/` and `build_g/` are configured
identically — diff their `CMakeCache.txt`: only a cosmetic `LLAMA_CURL` entry
differs). The mode is selected **at runtime** by environment variables plus the
contents of the file `layers2save` **in the process' current working
directory**. Selection logic: `tools/server/server.cpp:555-645`
(`llama_server()`).

| #     | name                 | python needed?       | what it does                                               |
| ----- | -------------------- | -------------------- | ---------------------------------------------------------- |
| **A** | plain / graph mode   | no                   | plain fast path; ladder optionally in-graph (`LADDER_BIN`) |
| **B** | python capture (shm) | yes (`xllamacpp.py`) | dumps `layers2save` tensors to shm; waits for python       |
| **C** | native C++ ladder    | no                   | same hook, ladder computed in C++ (no shm/python)          |

Callback functions (see §8): mode B = `detsoncb_share_activs`, mode C =
`detsoncb_ladder_cpp`.

There is **no 4th mode**. A "pure C++ capture to disk" (no python, no shm) is
**not implemented** in this tree. The only C++ capture-to-disk code ever
written is on the old remote branch `a/saveacts` (last commit 2025-11-26), in
`examples/eval-callback/eval-callback.cpp`: a standalone offline example
(`llama-eval-callback`, one prompt per run, no HTTP server) that appends
`int32 ne0, ne1` + raw f32 values to `activs.bin`. It is not merged, not
maintained, and its format is **not** what `b0.py:ChunkReader` reads. See §8 for
how to add it properly.

### Mode A — plain server / in-graph ladder

```bash
cd /home/xtof/git/llama.cpp
: > layers2save                      # keep it empty/absent (see gotchas)

# A0: plain server, no ladder (this is what xp_sqa*_base.py uses)
DETSON_OFF=1 ./build_g/bin/llama-server \
    -m /home/xtof/ggufs/Luciole-8B-Instruct-1.1-Q4_K_M.gguf \
    -ngl 99 -c 8192 --port 8082 --no-webui --no-warmup

# A1: same + ladder folded into the graph
~/envs/transformers/bin/python3 ../researchplm/ladder/cppladder_export_graph.py \
    --mlp logdir/xp_xxx_mlp.pt --out logdir/xp_xxx_graph.bin [--alpha 1.0]
DETSON_OFF=1 LADDER_BIN=/abs/path/xp_xxx_graph.bin LADDER_LAYER=19 \
    ./build_g/bin/llama-server -m MODEL.gguf -ngl 99 -c 8192 \
    --port 8082 --no-webui --no-warmup
```

* `LADDER_BIN` format (`cppladder_export_graph.py`): little-endian
  `int32 magic 'LADW' (0x4C414457)`, `int32 din = 2*n_embd`, `int32 hdim = n_embd`,
  then `f16 W[hdim*din]` with `W = alpha * vals @ keys^T / sqrt(din)`, so that
  `W @ rms_norm(x) == vals @ (keys . x/||x||)`.
* Loaded by `llama_model_base::load_ladder_weight()` (`src/llama-model.cpp:1834`),
  allocated on the **same backend buffer type as `output.weight`** (so it lands
  on the GPU), injected in the arch graph:
  * `src/models/nemotron-h.cpp:231` (after `build_norm(output_norm)`) and
    `:328` (capture of `l_out-K` in `build_ffn_layer`);
  * `src/models/qwen35.cpp:221` and `:206`.
  **Only these two architectures have the hook.** On any other arch the weight
  loads and the stderr line prints, but nothing is injected.
* `LADDER_LAYER` defaults to 19 (and falls back to 19 if out of range).
* Verify in stderr: `ladder: <path> (8192x4096 f16) on CUDA0, injected at l_out-19`.
* Verified equivalence: greedy outputs are token-identical to the mode-B/python
  ladder (f16 rounding flips no argmax); the ggml scheduler reuses the captured
  graph including the ladder ops.
* `DETSON_OFF` is **mandatory** here: without it the eval callback is installed
  unconditionally and the per-node launch/sync overhead makes prefill ~6x
  slower (see §6).

### Mode B — python capture via shared memory (the "detson" path)

```bash
cd /home/xtof/git/llama.cpp
printf 'l_out-19\nresult_norm\n' > layers2save   # trailing newline on every line!

LOGITS_ALL=1 PYTHONUNBUFFERED=1 \
  ~/envs/transformers/bin/python3 xllamacpp.py \
      --model MODEL.gguf --port 8261 --activs /abs/path/activs.bin
```

* `xllamacpp.py` is the **only** legal client: it starts the patched
  `./build/bin/llama-server` (hard-coded path, port **8257**, `-c 30000`,
  `--cache-type-k/v q8_0`, `--temp 0`, `-ngl 99` for dense models) and drains
  `/dev/shm/ring_buffer_demo` + `/dev/shm/sem.{c2py,py2c}_sem` from a python
  thread. It exposes an HTTP proxy on `--port` and optionally writes every
  dumped tensor to `--activs` (gzip chunks: `int32 ndim`, `int32[ndim] shape`,
  `int32 len(name)`, name, `int64 len(gz)`, gzip(f32 data)) — this is the format
  `b0.py:ChunkReader` and `train.py` read.
* `LOGITS_ALL=1` is required when you need rows for **every** prompt token: it
  sets the server output limits to `{n_batch, n_batch}`
  (`tools/server/server-context.cpp:43`), otherwise `output_reserve()` asserts
  as soon as a prefill wants logits for more tokens than `n_parallel*(1+n_draft)`.
* `layers2save` semantics (read from cwd at `server.cpp:596`):
  * every listed name is dumped (`detson_send_tensor`) when its node is
    computed;
  * **the LAST listed name is special**: after the python reply its tensor is
    overwritten in place from the shm buffer (`detsavelayer[i+1] == NULL`), so
    the python handler can modify the final embedding. Earlier entries are
    send-only.
  * each line is truncated by `line[strlen(line)-1]=0`, i.e. it strips the last
    character (assumed `\n`): **the file must end every line with a newline**,
    and an empty line terminates the list.
* Blocking contract: for every matched node the C++ side does `sem_post` then
  **`sem_wait`** — graph evaluation is frozen until python answers. A non-empty
  `layers2save` with **no** `xllamacpp.py` client = the server hangs forever.
* Shutdown: the server writes `424242.0f` in slot 0 and posts the semaphore;
  python sees the sentinel and stops. `xllamacpp.py` also `rm -f`s stale
  `/dev/shm/sem.*` on start.
* The listener thread is a **daemon** with a `stopped` Event: it acquires
  `/c2py_sem` with a 0.5 s timeout in a loop, so a SIGTERM-killed server (which
  never posts the sentinel) cannot wedge it, and every shutdown path does
  `stopped.set(); join(timeout=10)`.
* The OpenAI proxy **serializes** requests (`threading.Lock`): llama-server runs
  a single slot, so rollouts never interleave and the ladder pairing is exact.
  To run captures in parallel you must start several `xllamacpp.py` processes
  (different ports, and a distinct shm/sem name each — those names are
  hard-coded in both `xllamacpp.py` and `server.cpp`).
* Other env vars in this path: `TOKNOD=<node>` (also dump a matching *input*
  tensor of the node), `SAVE_EMB=<node>` (dump the unembedding matrix to
  `detembeds.bin`, requires `-ngl 0`), `NTOKS=<n>` (fixed `n_predict` in serve
  mode).
* Optional python-side ladder: `xllamacpp.py --ladder mlp.pt` installs
  `LadderHandler` (torch, per-token round trip through shm → **slow**, ~280
  ms/token on the Titan X). For speed use mode A or C instead.

### Mode C — native C++ ladder

```bash
cd /home/xtof/git/llama.cpp
printf 'l_out-19\nresult_norm\n' > layers2save
~/envs/transformers/bin/python3 ../researchplm/ladder/cppladder_export.py \
    --mlp logdir/xp_xxx_mlp.pt --out logdir/xp_xxx_cpp.bin

LADDER_CPP=/abs/path/xp_xxx_cpp.bin \
  ~/envs/transformers/bin/python3 xllamacpp.py --model MODEL.gguf --port 8261 ...
```

* Weights format (`cppladder_export.py`): `int32 magic 'LADR' (0x4C414452)`,
  `dhid, din, hdim, has_bias`, then f32 `keys[dhid*din]` (row-normalized),
  `vals[hdim*dhid]`, optional `vals_bias[hdim]`.
* The file is re-`stat()`ed at every `apply()` and **hot-reloaded** when its
  mtime changes (used by GRPO, where the policy changes between rollouts).
* Semantics = the fixed python handler: stash the last-token row of the first
  node (`l_out-19`), then at the second node (`result_norm`) add
  `vals @ (keys . xn)` with `xn = concat(prev, last)/||.||` to the **last-token
  row only**.
* `DETSON_OFF=1` may be set: it removes `cb_eval` only when it is *not*
  `detsoncb_ladder_cpp`, so mode C survives it.
* The `xllamacpp.py` in this tree must be the `cppladder` version: when
  `LADDER_CPP` **or `LADDER_GRAPH`** is set it uses `NoopHandler` (it prints
  `cppladder: ladder active natively in C++ (...) -> python handler skipped`),
  so the ladder is applied exactly once (not twice, python + C++).
  `LADDER_GRAPH` is for callers that go through `xllamacpp.py` while the ladder
  is folded in-graph via `LADDER_BIN`; it only prevents a double application
  (no script in `../researchplm` sets it today).
* Verify in stderr: `cppladder: native ladder installed from ... (nodes: ...)`
  and, with `#define CPPLADDER_TIMING 1` (default in `cppladder_ladder.h`),
  periodic `cppladder: cb(...) ... ms` lines.

### Runtime decision order (server.cpp:555-645)

1. Always (all modes): create `/dev/shm/ring_buffer_demo` (280 MB) and the two
   semaphores — even under `DETSON_OFF`.
2. `cb_eval = detsoncb_share_activs`, unless `SAVE_EMB` is set →
   `detsoncb_save_embeds`.
3. Read `layers2save` from **cwd** into `detsavelayer[]`.
4. If `LADDER_CPP` is set and `layers2save` lists ≥2 nodes → `cb_eval =
   detsoncb_ladder_cpp`.
5. If `DETSON_OFF` is set **and** `cb_eval != detsoncb_ladder_cpp` → `cb_eval =
   nullptr` (plain mode). Note step 2 installs the callback even when
   `layers2save` is empty — that is why plain runs must set `DETSON_OFF`.

## 3. Environment variable reference

| var                 | mode | effect                                                        |
| ------------------- | ---- | ------------------------------------------------------------- |
| `DETSON_OFF=1`      | A    | remove the eval callback → plain fast path (no per-node sync) |
| `LADDER_BIN=<path>` | A    | f16 folded ladder matrix injected in-graph (`'LADW'`)         |
| `LADDER_LAYER=<n>`  | A    | block whose `l_out-n` feeds the ladder (default 19)           |
| `LADDER_CPP=<path>` | C    | native C++ ladder weights (`'LADR'`, hot-reload)              |
| `LOGITS_ALL=1`      | B    | request logits/rows for every prefill token                   |
| `TOKNOD=<node>`     | B    | additionally dump a matching input tensor                     |
| `SAVE_EMB=<node>`   | B    | dump the unembedding matrix to `detembeds.bin` (`-ngl 0`)     |
| `NTOKS=<n>`         | B    | fixed `n_predict` in `xllamacpp.py` serve mode                |
| `LADDER_GRAPH`      | A    | python-side flag → `xllamacpp.py` uses `NoopHandler`          |

## 4. Build layout and rebuild rules

* `build/` — the capture build; **hard-coded** in `xllamacpp.py`
  (`llamacppdir + "/build/bin/llama-server"`). Older than `build_g/`.
* `build_g/` — the graph-mode build used by `b0.py:GraphServer`
  (`LLAMACPP + "build_g/bin/llama-server"`).
* Both are thin executables (~17 KB): the code lives in `libllama.so`,
  `libggml*.so` **inside the same build dir**, so the `.so` of a dir must be
  relinked together with its binary.
* **Never relink a build dir whose server is running**: GNU ld truncates output
  files in place, which corrupts the live process. Build in the other dir (or
  stop the server). Also restart the server after relinking — an already
  running one keeps the old `.so`.
* Typical commands:
  `cmake -B build -DGGML_CUDA=ON` then `cmake --build build --target llama-server -j`.
  On Jean Zay: `module load arch/h100` first; keep a second dir for safe
  rebuilds.
* The CUDA build is what matters (`-ngl 99`); the CPU-only path is untested here.

## 5. `layers2save` — the cwd trap

* It is read from the **process current working directory**, not from the build
  dir, and **`xllamacpp.py` only reads it** (it uses `layers[-1]` to know which
  node the python handler may modify). The *caller* writes it before launching
  the server: `b0.py:CaptureServer.start()` writes the two node names,
  `GraphServer.start()` truncates it, `xp_sqa*_base.py` truncates it. Never
  launch a mode-A server from a cwd where a stale non-empty `layers2save`
  exists **without** `DETSON_OFF`.
* Node names come from `cb(cur, "l_out", il)` (→ `l_out-19`) and
  `cb(cur, "result_norm", -1)` in `src/models/*.cpp`.

## 6. Performance notes (measured on lully, Titan X Pascal, 8B Q4_K_M)

| path                                     | rate                           | note                             |
| ---------------------------------------- | ------------------------------ | -------------------------------- |
| graph decode (`DETSON_OFF`+`LADDER_BIN`) | **40.7 tok/s** (24.6 ms/token) | bandwidth-bound                  |
| graph prefill                            | **350 tok/s** (2.86 ms/token)  |                                  |
| mode B capture prefill                   | **53 tok/s** (18.7 ms/token)   | ~6.5x slower than graph prefill  |
| python `LadderHandler`                   | ~280 ms/token                  | shm + 2 semaphores + interpreter |
| native C++ ladder (9B, other box)        | ~145 ms/token                  | ~100 ms is callback infra        |

The mode-B slowdown is **not** the python/gzip work: it comes from
`detsoncb_share_activs` returning `true` for **every** node in the `ask` phase
(`server.cpp:122`), which makes `ggml_backend_sched` launch **and device-sync
every node separately**. Measured: 3.7 s for a ~200-token prefill vs 0.6 s in
graph mode. The fix is the one already used by `detsoncb_ladder_cpp`: in the
`ask` phase answer `true` **only** for the `layers2save` nodes (and for
`TOKNOD`'s tensor), `false` otherwise — then all other nodes are grouped into
one subgraph launch. Guard it with an env var so `TOKNOD`/`SAVE_EMB` users are
not silently broken.

## 7. Mode B internals: the shared-memory handshake (the "detson" hook)

Everything below is mode B only (python capture / python `LadderHandler`).

**Where the hook sits.** `params.cb_eval` is handed to
`ggml_backend_sched_set_eval_callback` (`src/llama-context.cpp:1362`, API at
`ggml/src/ggml-backend.cpp:1991`). In `ggml_backend_sched_graph_compute_async`
(`ggml/src/ggml-backend.cpp:1744-1778`) the scheduler walks the nodes of each
backend split:

```c
bool need = sched->callback_eval(t, true, ud);   // "do you want this node?"
// group consecutive nodes with need == false ...
ggml_backend_graph_compute_async(split_backend, &gv);
ggml_backend_synchronize(split_backend);         // always
if (need && !sched->callback_eval(t, false, ud)) break;   // the eval phase
```

So the `ask=false` phase runs **after the node's group is computed and the
device is synchronized, and before its consumers are computed** — exactly the
window where in-place tampering is effective. There is **no re-injection op in
the graph**: the modified values are simply already in the tensor's buffer when
the downstream ops read it (residual add, next blocks, `output.weight` mul-mat).

**Out path (C++ → python).** `detson_send_tensor` (`server.cpp:65`) serializes
into `/dev/shm/ring_buffer_demo`:

```
float[100]  node name (NUL-padded ASCII)
float       ne1
float       ne0
float[]     ne3*ne2*ne1*ne0 values, converted to f32 (f16/i32/i16/i8 supported)
```

The segment is 70 M floats (280 MB, `float buffers[1][70000000]`, sized for
ubatch 16384 x 4096), created with `shm_open`+`ftruncate`+`mmap`
(`server.cpp:562-563`) at server start. Then `sem_post(/c2py_sem)` and
**`sem_wait(/py2c_sem)`**: graph evaluation is frozen until python answers
(stop-the-world). One tensor at a time, buffer reused. Constraint: a hooked
tensor must have fewer than 70 M elements (e.g. `n_tokens * n_embd < 70M`).

**In path (python → C++).** The python side (`xllamacpp.py:SharedMem`) maps the
same file and may overwrite the buffer (for the ladder: only the last row, via
`write_last_token`, at byte offset `4*(102 + (ne1-1)*ne0)`). Back in
`detsoncb_share_activs` the tensor is copied back **in place** iff

1. `shm->buffers[0][0] != 424242.0f` (not the shutdown sentinel — by default
   slot 0 holds the first name byte, so the copy-back is active), and
2. the node is the **last** entry of `detsavelayer` (i.e. the last line of
   `layers2save`); earlier entries are send-only.

Host buffers are written with `memcpy` into `t->data`; device buffers with
`ggml_backend_tensor_set` (and read with `ggml_backend_tensor_get` into
`cb_data->data`), so it works with `-ngl 99`.

**Shutdown.** On quit the server writes `424242.0f` into slot 0 and posts
`/c2py_sem` (`server.cpp:989`); python sees the sentinel, stops, and removes
`/dev/shm/sem.*`. A leftover `ring_buffer_demo` is harmless (it is
`ftruncate`d on reuse) but a leftover non-empty `layers2save` is not — see §5.

**Cost model.** With `ask=true` for every node (§6) every group is a single
node: ~500 launches + ~500 device syncs per forward pass. Python's gzip/numpy
work is negligible in comparison (2 activations per prefill, not per token).

## 8. Where the code is

| file                                 | content                                                |
| ------------------------------------ | ------------------------------------------------------ |
| `tools/server/server.cpp:555-645`    | detson block: shm/sems, `layers2save`, env selection   |
| `tools/server/server.cpp:121`        | `detsoncb_share_activs` (mode B hook)                  |
| `tools/server/server.cpp:339`        | `detsoncb_ladder_cpp` (mode C hook)                    |
| `tools/server/server.cpp:989`        | shutdown sentinel `424242.0f`                          |
| `tools/server/cppladder_ladder.h`    | `LadderCpp`: load/store/apply, hot-reload              |
| `src/llama-model.cpp:1834`           | `load_ladder_weight()` (`LADDER_BIN`)                  |
| `src/llama-model.h:640`              | `ladder_w`, `ladder_ctx`, `ladder_buf`, `ladder_layer` |
| `src/models/nemotron-h.cpp:231,328`  | in-graph ladder + `l_out-K` capture                    |
| `src/models/qwen35.cpp:221,206`      | idem for qwen35                                        |
| `tools/server/server-context.cpp:43` | `LOGITS_ALL` output limits                             |
| `xllamacpp.py`                       | shm client, `ActivsSaver`, OpenAI proxy (`/shutdown`)  |
| `README.xtof.md`                     | layman/user guide (3 modes + commands)                 |

## 9. Known issues / possible next steps

* **Mode-B `ask` filter** (§6): biggest single win, ~6x on capture, ~40 lines
  guarded by an env var. Would also let capture reach graph-mode prefill speed.
* **Pure-C++ capture to disk (no python)**: not implemented. Sketch: after the
  `ask` filter, in the `ask=false` branch of the two matched nodes append a
  chunk to a file directly in C++ using the same format as `ActivsSaver`
  (`int32 ndim`, `int32[ndim] shape`, `int32 len(name)`, name, `int64 len(gz)`,
  gzip(f32)) — or raw f32 plus a sidecar index. That removes shm, semaphores,
  `xllamacpp.py`, `posix_ipc` and the python thread from the capture path.
  Do not reuse the old `a/saveacts` format (raw `ne0, ne1` + f32, no names).
* **Graph-mode ladder is only wired for `nemotron_h` and `qwen35`.** Any other
  arch needs its own two-line patch (capture `cur` at `l_out-K`, add
  `ladder_w @ rms_norm(concat(...))` after `output_norm`).
* `tools/server/cppladder_server.cpp` is a stale copy of `server.cpp`; keep it
  in sync (or delete it) — `cppladder_gen.py` regenerates it from the current
  `server.cpp`.
* `LADDER_CPP` and `LADDER_BIN` are independent: setting both is legal but the
  in-graph one wins on the archs that support it (the callback would then also
  apply a second ladder — do not do it).
* Stale `/dev/shm/ring_buffer_demo` and `/dev/shm/sem.*` survive crashes;
  `xllamacpp.py` cleans the semaphores, the shm segment is `ftruncate`d on
  reuse. If a server seems deadlocked at startup, check for a leftover client.
