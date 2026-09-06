# Reads a log file produced by the ActivsSaver debug output in xllamacpp.py
# (lines starting with "[stderr] node: <name>"). From the "node:" list it:
#   - collects the layer-output nodes (ending a decoder layer), any line whose
#     node name starts with "l_out" or "l-out"
#   - derives the total number of layers from the largest layer index seen
#   - picks the two nodes whose activations we want to monitor and writes their
#     names (one per line) to the file "layers2save":
#       * the layer output at about 60% of the network depth after layer 0
#       * the final hidden state, i.e. the node just before the unembedding
# Usage:
#   python init_layers.py <log_file>

# TODO: also find the node which dimensions are all < 100k but which has one of
# his sources with one dim > 100k: this node computes the input embedding vector
# and you must save the node name as well as its source number with dim >100k
# so that I can later on print the tokens index with a special callback on this
# source of this node

import re
import sys


def main():
    if len(sys.argv) < 2:
        print("arg1 = name of log file from SAVE_EMB")
        sys.exit(1)

    logfile = sys.argv[1]

    # one node name per line, in graph order; the "saving embeddings" line is
    # printed right after the node holding the unembedding matrix, so that node
    # (the final one to save) is the one just before it
    nodes = []
    node_final = None
    with open(logfile, "r") as f:
        for line in f:
            if line.startswith("[stderr] node: "):
                nodes.append(line[len("[stderr] node: "):].strip())
            elif "saving embeddings" in line and nodes:
                node_final = nodes[-1]

    # layer-end nodes: keep each layer index, remember its node name
    # name pattern is "<prefix><index>" (e.g. "l_out-31")
    layer_nodes = {}
    for name in nodes:
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


if __name__ == "__main__":
    main()
