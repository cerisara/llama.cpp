# Shared-memory activation handoff during decode (`sharedram` branch)

This notes how the changes by `xtof` on the `sharedram` branch (commits
`fe175ff16` → `86eb532c6`, all in `tools/server/server.cpp`) enable a shared
memory exchange with an external program (Python) during decode: activations
are sent out, modified externally, and the modified values are reinjected into
the decoding pass.

## The short answer

There is **no dedicated re-injection point in the graph** — the modified
values are written back **in place** into the live intermediate tensor's data
buffer, *between the compute of that layer's node and the compute of its
downstream consumers*. The rest of the decode pass simply reads the modified
data; no graph surgery or explicit reinjection op is involved.

## How it hangs together

### 1. Hook: the ggml scheduler eval-callback

- `params.cb_eval = detsoncb_share_activs` (`server.cpp:373`) is passed to
  `ggml_backend_sched_set_eval_callback` (`src/llama-context.cpp:779`,
  upstream API declared at `ggml/include/ggml-backend.h:303`).
- The scheduler invokes the callback per node in topological order:
  `ask = true` before a chunk of nodes is computed, then `ask = false` **after**
  the chunk containing the node has been computed **and**
  `ggml_backend_synchronize` has been called (`ggml/src/ggml-backend.cpp:1558-1595`).
  So by the time the `ask = false` branch runs, the tensor is fresh in its
  buffer and its consumers have **not** run yet — the callback sits exactly in
  the window where in-place tampering is effective.

### 2. Out path (C++ → Python)

- `detson_send_tensor` (`server.cpp:43`) serializes the tensor —
  first 2 floats = dims, then all values as f32 — into the POSIX shm segment
  `/ring_buffer_demo` (10 M floats, created with `shm_open` + `ftruncate` +
  `mmap` at `server.cpp:325-328`), then posts `/c2py_sem` and **blocks** on
  `sem_wait(&sem_py2c)` (`server.cpp:89-90`).
- Graph evaluation is frozen until Python replies: it is a fully synchronous
  (stop-the-world) handoff during decode.

### 3. Re-injection (Python → C++)

In `detsoncb_share_activs` (`server.cpp:247-265`), after the semaphore
round-trip, the modified values are copied back if two conditions hold:

1. Python set `shm->buffers[0][0] = 424242.0f` — its "I modified it" flag.
   If unset, the original activations pass through untouched.
2. The tensor is the **last entry** in `detsavelayer`
   (`detsavelayer[i+1] == NULL`). Those names come from the runtime file
   `layers2save`; earlier listed layers are send-only.

The write-back re-reads the shm buffer element-by-element in the same index
order and overwrites the tensor in place:

```c
// recopie la shared RAM dans le computation graph de llamacpp
uint8_t * data = (uint8_t *) t->data;
...
float py_val = shm->buffers[0][bufidx++];
if (t->type == GGML_TYPE_F16)  *v = ggml_fp32_to_fp16(py_val);
else if (t->type == GGML_TYPE_F32) *v = py_val;
```

The next ops in the graph (residual add, the following blocks, and ultimately
the final `MUL_MAT` against `output.weight` / unembedding) then execute
normally and read the modified data. Debug commit `2532fd765` confirmed the
`output.weight` mul-mat produces the same next token as implied by the
modified activations — consistent with pure in-place reinjection.

## Caveats

- **Host-visible memory only**: the write-back uses `t->data` directly,
  regardless of where the tensor lives. For non-host (e.g. GPU) buffers that's
  a device pointer; the send path copies to `cb_data->data` for device
  tensors, but the re-injection path does not. So it only works when the
  hooked tensor's buffer is host-visible (or UMA).
- **Stop-the-world cost**: the callback runs synchronously inside the
  scheduler loop, so each hooked layer stalls the whole decode until Python
  round-trips the ~40 MB segment.

## Key locations

| What | Where |
|---|---|
| SHM name, sem names, `SharedMemory` struct | `tools/server/server.cpp:28-36` |
| shm creation (`shm_open`/`mmap`) | `tools/server/server.cpp:325-328` |
| Out path `detson_send_tensor` + sem handshake | `tools/server/server.cpp:43-91` |
| Callback `detsoncb_share_activs` | `tools/server/server.cpp:~106` |
| In-place re-injection loop | `tools/server/server.cpp:247-265` |
| Callback registration (`cb_eval`) | `tools/server/server.cpp:370-375` |
| `cb_eval` plumbing | `common/common.h:333`, `src/llama-context.cpp:779`, `include/llama.h:343` |
| Callback invocation in scheduler | `ggml/src/ggml-backend.cpp:1558-1595` |
