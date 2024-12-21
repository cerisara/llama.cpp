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

static void zeros(std::ofstream & file, size_t n) {
    char zero = 0;
    for (size_t i = 0; i < n; ++i) {
        file.write(&zero, 1);
    }
}

void print() {
    struct ggml_context * ctx_meta = NULL;
    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &ctx_meta,
    };
    std::vector<uint8_t> read_data;

    auto * ctx_gguf = gguf_init_from_file("/mnt/dos/xtof/gguf_ggml_models/qwen2.5-1.5b-instruct-q4_k_m.gguf", params);

    auto n_tensors = gguf_get_n_tensors(ctx_gguf);
    for (int i_tensor = 0; i_tensor < n_tensors; i_tensor++) {
        const char * t_name = gguf_get_tensor_name(ctx_gguf, i_tensor);
        struct ggml_tensor * t = ggml_get_tensor(ctx_meta, t_name);
        printf("tensor %s %d %d\n", t->name, t->ne[0], t->ne[1]);
    }
}

void copy() {
    struct ggml_context * ctx_meta = NULL;
    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &ctx_meta,
    };
    std::vector<uint8_t> read_data;

    // struct gguf_context * ctx_gguf;
    auto * ctx_gguf = gguf_init_from_file("/mnt/dos/xtof/gguf_ggml_models/qwen2.5-1.5b-instruct-q4_k_m.gguf", params);
    auto * ctx_out = gguf_init_empty();

    std::ofstream fout("tmp.gguf", std::ios::binary);
    fout.exceptions(std::ofstream::failbit); // fail fast on write errors
    gguf_set_kv(ctx_out, ctx_gguf);

    auto n_tensors = gguf_get_n_tensors(ctx_gguf);
    for (int i_tensor = 0; i_tensor < n_tensors; i_tensor++) {
        const char * t_name = gguf_get_tensor_name(ctx_gguf, i_tensor);
        struct ggml_tensor * t = ggml_get_tensor(ctx_meta, t_name);
        printf("tensor %s %d %d\n", t->name, t->ne[0], t->ne[1]);
        gguf_add_tensor(ctx_out, t);
    }

    // placeholder for the meta data
    {
        auto meta_size = gguf_get_meta_size(ctx_out);
        ::zeros(fout, meta_size);
    }

    // Write tensors data
    std::ifstream f_input("/mnt/dos/xtof/gguf_ggml_models/qwen2.5-1.5b-instruct-q4_k_m.gguf", std::ios::binary);
    if (!f_input.is_open()) {
        fprintf(stderr, "%s:  failed to open input GGUF \n", __func__);
        gguf_free(ctx_gguf);
        gguf_free(ctx_out);
        fout.close();
        exit(EXIT_FAILURE);
    }

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

int main(int argc, const char ** argv) {
    // copy();
    print();
}
