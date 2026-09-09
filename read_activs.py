# Reads back the activation file written by ActivsSaver in xllamacpp.py
# Usage:
#   python read_activs.py <file>                 # summary of all chunks
#   python read_activs.py <file> <idx2save> [<dim>]  # show first 5 values of the given tensor
#                 <dim>: slice along first dim; defaults to the last vector
#   python read_activs.py <file> --save out.npz  # dump all chunks to a .npz file

import sys
import gzip

import numpy as np


def load_activs(infile):
    # reads back the chunks in order (same format as ActivsSaver writes)
    activs = []
    names = []
    with open(infile, "rb") as f:
        while True:
            head = f.read(4)
            if len(head) == 0:
                break
            ndim = np.frombuffer(head, dtype=np.int32).item()
            shape = np.frombuffer(f.read(4 * ndim), dtype=np.int32).tolist()
            namelen = np.frombuffer(f.read(4), dtype=np.int32).item()
            name = f.read(namelen).decode('ascii', errors='replace')
            names.append(name)
            nbytes = np.frombuffer(f.read(8), dtype=np.int64).item()
            data = np.frombuffer(gzip.decompress(f.read(nbytes)), dtype=np.float32)
            data.shape = shape
            activs.append(data)
    return activs, names


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    infile = sys.argv[1]
    idx2save = None
    dim = None
    outfile = None
    vec = None
    toks = False
    args = sys.argv[2:]
    while args:
        if args[0] == "--save":
            outfile = args[1]
            args = args[2:]
        elif args[0] == "--vec":
            vec = int(args[1])
            args = args[2:]
        elif args[0] == "--tokens":
            toks = True
            args = args[2:]
        else:
            # positional args: tensor index, then first-dim slice
            try:
                if idx2save is None:
                    idx2save = int(args[0])
                elif dim is None:
                    dim = int(args[0])
                else:
                    print("ignoring unknown argument: " + args[0])
            except ValueError:
                print("ignoring unknown argument: " + args[0])
            args = args[1:]

    print("reading " + infile)
    activs, names = load_activs(infile)
    print("loaded " + str(len(activs)) + " activation tensors")

    if outfile is not None:
        np.savez_compressed(outfile, *activs)
        print("dumped all chunks to " + outfile)

    if toks:
        # assume the first node is a token node
        # token nodes are shape (1, n_tokens): flatten to read the ids
        tokens = []
        for v in range(len(names)):
            node = names[v]
            if node == names[0]:
                tokens.extend([int(i) for i in activs[v].flatten()])
        print("TOKENS", " ".join([str(t) for t in tokens]))
        return

    if vec is not None:
        if vec < 0 or vec >= len(activs):
            print("index " + str(vec) + " out of range")
            sys.exit(1)
        arr = activs[vec]
        node = names[vec] if vec < len(names) else ""
        print("tensor " + str(vec) + ": " + node + " shape=" + str(arr.shape))
        print("VEC"," ".join([str(x) for x in arr.flatten()]))
        return

    if idx2save is not None:
        if idx2save < 0 or idx2save >= len(activs):
            print("idx2save " + str(idx2save) + " out of range")
            sys.exit(1)
        arr = activs[idx2save]
        if arr.size == 0:
            print("tensor " + str(idx2save) + " shape=" + str(arr.shape) + " is empty, skipping")
            return
        default_dim = arr.shape[0] - 1
        d = dim if dim is not None else default_dim
        if d < 0 or d >= arr.shape[0]:
            print("dim " + str(d) + " out of range (0.." + str(arr.shape[0] - 1) + ")")
            sys.exit(1)
        print("tensor " + str(idx2save) + " shape=" + str(arr.shape) + " dim=" + str(d))
        arr = arr[d]
        arr.tofile("onevec.bin")
        print("saved vector to onevec.bin")
        print("first 5 values: " + " ".join([str(x) for x in arr.flatten()[:5]]))
        return

    # summary of every chunk
    for i, arr in enumerate(activs):
        node = names[i] if i < len(names) else ""
        if arr.size == 0:
            print(str(i) + ": " + node + " shape=" + str(arr.shape) + " (empty)")
        else:
            print(str(i) + ": " + node + " shape=" + str(arr.shape) + " min=" + str(arr.min()) + " max=" + str(arr.max()))


if __name__ == "__main__":
    main()
