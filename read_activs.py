# Reads back the activation file written by ActivsSaver in xllamacpp.py
# Usage:
#   python read_activs.py <file>                 # summary of all chunks
#   python read_activs.py <file> -i 5            # show first values of chunk 5
#   python read_activs.py <file> --save out.npz  # dump all chunks to a .npz file

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
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    infile = sys.argv[1]
    index = None
    outfile = None
    args = sys.argv[2:]
    while args:
        if args[0] == "-i":
            index = int(args[1])
            args = args[2:]
        elif args[0] == "--save":
            outfile = args[1]
            args = args[2:]
        else:
            args = args[1:]

    print("reading " + infile)
    activs = load_activs(infile)
    print("loaded " + str(len(activs)) + " activation tensors")

    if outfile is not None:
        np.savez_compressed(outfile, *activs)
        print("dumped all chunks to " + outfile)

    if index is not None:
        if index < 0 or index >= len(activs):
            print("index " + str(index) + " out of range")
            sys.exit(1)
        arr = activs[index]
        print("chunk " + str(index) + " shape=" + str(arr.shape))
        print("first 10 values: " + " ".join([str(x) for x in arr.flatten()[:10]]))
        print("min=" + str(arr.min()) + " max=" + str(arr.max()) + " mean=" + str(arr.mean()))
        return

    # summary of every chunk
    for i, arr in enumerate(activs):
        print(str(i) + ": shape=" + str(arr.shape) + " min=" + str(arr.min()) + " max=" + str(arr.max()))


if __name__ == "__main__":
    main()
