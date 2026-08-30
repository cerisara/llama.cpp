# Compares two activation files written by ActivsSaver in xllamacpp.py.
# For every chunk (layer) and every time step, prints the magnitude (L2 norm)
# of the difference between the two files.
# Even when the prefill tensors of the two files have different shapes, the
# 1-dim vector of each time step is extracted and matched against the
# corresponding time step of the other file.
# Usage:
#   python diff_activs.py <file_a> <file_b>
#   python diff_activs.py <file_a> <file_b> --save out.npz  # dump diffs to .npz

import sys
import gzip

import numpy as np


def load_activs(infile):
    # reads back the chunks in order (same format as ActivsSaver writes)
    activs = []
    with open(infile, "rb") as f:
        while True:
            head = f.read(4)
            if len(head) == 0:
                break
            ndim = np.frombuffer(head, dtype=np.int32).item()
            shape = np.frombuffer(f.read(4 * ndim), dtype=np.int32).tolist()
            nbytes = np.frombuffer(f.read(8), dtype=np.int64).item()
            data = np.frombuffer(gzip.decompress(f.read(nbytes)), dtype=np.float32)
            data.shape = shape
            activs.append(data)
    return activs


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    infile_a = sys.argv[1]
    infile_b = sys.argv[2]
    outfile = None
    args = sys.argv[3:]
    while args:
        if args[0] == "--save":
            outfile = args[1]
            args = args[2:]
        else:
            args = args[1:]

    print("reading " + infile_a)
    activs_a = load_activs(infile_a)
    print("loaded " + str(len(activs_a)) + " activation tensors")

    print("reading " + infile_b)
    activs_b = load_activs(infile_b)
    print("loaded " + str(len(activs_b)) + " activation tensors")

    if len(activs_a) != len(activs_b):
        print("files have different number of tensors: " + str(len(activs_a)) + " vs " + str(len(activs_b)))
        sys.exit(1)

    diffs = []
    ft = 0
    for i, (arr_a, arr_b) in enumerate(zip(activs_a, activs_b)):
        # extract the 1-dim vector of every time step (last axis is hidden size)
        # the prefill tensor may hold many tokens at once, so decompose it
        # into individual time steps first
        vecs_a = [v for v in arr_a.reshape(-1, arr_a.shape[-1])]
        vecs_b = [v for v in arr_b.reshape(-1, arr_b.shape[-1])]

        # pair steps positionally, even across different prefill/generation splits
        n = min(len(vecs_a), len(vecs_b))
        if len(vecs_a) != len(vecs_b):
            print("tensor " + str(i) + ": step counts differ (" + str(len(vecs_a)) + " vs " + str(len(vecs_b)) + "), comparing first " + str(n))

        diff = np.zeros(n)
        for t in range(n):
            diff[t] = np.linalg.norm(vecs_a[t] - vecs_b[t])
        diffs.append(diff)

        for u in range(n):
            print("diff t=" + str(ft) + " diff_norm=" + str(diff[u]))
            ft = ft+1

    if outfile is not None:
        np.savez_compressed(outfile, *diffs)
        print("dumped diffs to " + outfile)


if __name__ == "__main__":
    main()
