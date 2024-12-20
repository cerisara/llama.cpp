#include "llama.h"
#include "common.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include <stdio.h>
#include <string.h>
#include <climits>
#include <stdexcept>

int main(int argc, const char ** argv) {
    struct ggml_context * ctx_meta = NULL;
    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &ctx_meta,
    };

    struct gguf_context * ctx_gguf;
    auto * ctx_gguf = gguf_init_from_file("mod.gguf", params);
    auto * ctx_out = gguf_init_empty();

    std::ofstream fout(split_params.output.c_str(), std::ios::binary);
    fout.exceptions(std::ofstream::failbit); // fail fast on write errors

    auto n_tensors = gguf_get_n_tensors(ctx_gguf);
    for (int i_tensor = 0; i_tensor < n_tensors; i_tensor++) {
        const char * t_name = gguf_get_tensor_name(ctx_gguf, i_tensor);
        struct ggml_tensor * t = ggml_get_tensor(ctx_meta, t_name);
        gguf_add_tensor(ctx_out, t);
    }

    // placeholder for the meta data
    {
        auto meta_size = gguf_get_meta_size(ctx_out);
        ::zeros(fout, meta_size);
    }

    // Write tensors data
    std::ifstream f_input(split_path, std::ios::binary);
    if (!f_input.is_open()) {
        fprintf(stderr, "%s:  failed to open input GGUF from %s\n", __func__, split_path);
        for (uint32_t i = 0; i < ctx_ggufs.size(); i++) {
            gguf_free(ctx_ggufs[i]);
            ggml_free(ctx_metas[i]);
        }
        gguf_free(ctx_out);
        fout.close();
        exit(EXIT_FAILURE);
    }
    fprintf(stderr, "%s: writing tensors %s ...", __func__, split_path);

    auto * ctx_gguf = ctx_ggufs[i_split];
    auto * ctx_meta = ctx_metas[i_split];

    auto n_tensors = gguf_get_n_tensors(ctx_gguf);
    for (int i_tensor = 0; i_tensor < n_tensors; i_tensor++) {
        const char * t_name = gguf_get_tensor_name(ctx_gguf, i_tensor);
        struct ggml_tensor * t = ggml_get_tensor(ctx_meta, t_name);

        auto n_bytes = ggml_nbytes(t);

        if (read_data.size() < n_bytes) {
            read_data.resize(n_bytes);
        }

        auto offset = gguf_get_data_offset(ctx_gguf) + gguf_get_tensor_offset(ctx_gguf, i_tensor);
        f_input.seekg(offset);
        f_input.read((char *)read_data.data(), n_bytes);

        // write tensor data + padding
        fout.write((const char *)read_data.data(), n_bytes);
        zeros(fout, GGML_PAD(n_bytes, GGUF_DEFAULT_ALIGNMENT) - n_bytes);
    }

    gguf_free(ctx_gguf);
    ggml_free(ctx_meta);
    f_input.close();
    fprintf(stderr, "\033[3Ddone\n");

    {
        // go back to beginning of file and write the updated metadata
        fout.seekp(0);
        std::vector<uint8_t> data(gguf_get_meta_size(ctx_out));
        gguf_get_meta_data(ctx_out, data.data());
        fout.write((const char *)data.data(), data.size());

        fout.close();
        gguf_free(ctx_out);
    }


}

