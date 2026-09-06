# Reads a log file produced by the ActivsSaver debug output in xllamacpp.py
# (lines starting with "[stderr] node: <name>"). From the "node:" list it:
#   - collects the layer-output nodes (ending a decoder layer), any line whose
#     node name starts with "l_out" or "l-out"
#   - derives the total number of layers from the largest layer index seen
#   - picks the two nodes whose activations we want to monitor and writes their
#     names (one per line) to the file "layers2save":
#       * the layer output at about 60% of the network depth after layer 0
#       * the final hidden state, i.e. the node just before the unembedding
#   - writes to "toknod.txt" the input-embedding node name and the number of
#     its source that holds the vocab dimension (node dims all < 100k but one
#     source with a dim > 100k), for a callback to print the tokens index
# Usage:
#   python init_layers.py <log_file>

import re
import sys

# a node with all its own dims < this but a source with one dim > this is the
# input-embedding node (its weight holds the vocab dimension)
TOKLIMIT = 100000


def parse_dims(s):
    # "896x2x1x1" -> [896, 2, 1, 1]
    return [int(x) for x in s.split("x")]


def _trim_dims(toks):
    # toks are the space-separated tokens of one node or source; the trailing
    # dims token matches NxNxNxN, anything before it (may be empty, or a name
    # with a suffix like "(reshaped)") is the name
    for i in range(len(toks) - 1, -1, -1):
        if re.fullmatch(r"\d+x\d+x\d+x\d+", toks[i]):
            return " ".join(toks[:i]), parse_dims(toks[i])
    return " ".join(toks), []


def parse_node_line(s):
    # line after "[stderr] node: ":
    #   "<name> <dims> <- <src0name> <src0dims> <- <src1name> <src1dims>"
    # returns (name, dims, [(srcname, srcdims), ...])
    parts = s.split(" <- ")
    name, dims = _trim_dims(parts[0].split(" "))
    srcs = []
    for p in parts[1:]:
        srcs.append(_trim_dims(p.split(" ")))
    return name, dims, srcs


def main():
    if len(sys.argv) < 2:
        print("arg1 = name of log file from SAVE_EMB")
        sys.exit(1)

    logfile = sys.argv[1]

    # one node per line, in graph order; the "saving embeddings" line is
    # printed right after the node holding the unembedding matrix, so that node
    # (the final one to save) is the one just before it
    nodes = []          # parsed (name, dims, srcs)
    node_final = None
    tok_name = None     # input-embedding node
    tok_src = None      # index of its source holding the vocab dimension
    with open(logfile, "r", errors="replace") as f:
        for line in f:
            if line.startswith("[stderr] node: "):
                name, dims, srcs = parse_node_line(line[len("[stderr] node: "):].strip())
                nodes.append((name, dims, srcs))
                # the input-embedding node: all its own dims < TOKLIMIT but one
                # of its sources has a dim > TOKLIMIT (the vocab). remember the
                # node name and the source number so a callback can later print
                # the tokens index from that source
                if tok_name is None and dims and all(d < TOKLIMIT for d in dims):
                    for si, sdims in srcs:
                        if any(d > TOKLIMIT for d in sdims):
                            tok_name = name
                            tok_src = si
                            break
            elif "saving embeddings" in line and nodes:
                node_final = nodes[-1][0]

    # layer-end nodes: keep each layer index, remember its node name
    # name pattern is "<prefix><index>" (e.g. "l_out-31")
    layer_nodes = {}
    for name, _, _ in nodes:
        m = re.match(r"^(l[-_]out)-?(\d+)$", name)
        if m:
            layer_nodes[int(m.group(2))] = name

    if not layer_nodes:
        print("no layer-output node found in " + logfile)
        sys.exit(1)

    n_layers = max(layer_nodes) + 1

    # node at about 60% of the depth after layer 0
    depth_idx = int(round(0.6 * (n_layers - 1)))
    node_60 = layer_nodes[depth_idx]

    if node_final is None:
        print("no 'saving embeddings' marker found, cannot locate final node")
        sys.exit(1)

    print("found " + str(n_layers) + " layers")
    print("60% node:   " + node_60)
    print("final node: " + node_final)

    with open("layers2save", "w") as f:
        f.write(node_60 + "\n")
        f.write(node_final + "\n")

    print("wrote nodes to layers2save")

    if tok_name is None:
        print("no input-embedding node found in " + logfile)
        sys.exit(1)

    print("tok node: " + tok_name + " source " + str(tok_src))

    with open("toknod.txt", "w") as f:
        f.write(str(tok_src) + "\n")

    print("wrote nodes to toknod.txt")


if __name__ == "__main__":
    main()
