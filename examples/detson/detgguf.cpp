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

    // auto * ctx_gguf = gguf_init_from_file("/mnt/dos/xtof/gguf_ggml_models/qwen2.5-1.5b-instruct-q4_k_m.gguf", params);
    auto * ctx_gguf = gguf_init_from_file("tmp.gguf", params);

    auto n_tensors = gguf_get_n_tensors(ctx_gguf);
	printf("ntensors %d\n",n_tensors);
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
	char *buf = (char *)malloc(500000000);

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

	// je reconstruis ctx_out pour modifier les dims des nouveaux tensors
	gguf_free(ctx_out);
	ctx_out = gguf_init_empty();
    gguf_set_kv(ctx_out, ctx_gguf);

    for (int i_tensor = 0; i_tensor < n_tensors; i_tensor++) {
        const char * t_name = gguf_get_tensor_name(ctx_gguf, i_tensor);
        struct ggml_tensor * t = ggml_get_tensor(ctx_meta, t_name);

        auto n_bytes = ggml_nbytes(t);

        auto offset = gguf_get_data_offset(ctx_gguf) + gguf_get_tensor_offset(ctx_gguf, i_tensor);
        f_input.seekg(offset);
		printf("tens %d %d\n",i_tensor, n_bytes);
        f_input.read(buf, n_bytes);
    
        char *nom = t->name;
        int npts=0, j=-1;
        for (int i=0;i<strlen(nom);i++) {
            if (nom[i]=='.') {
                npts++;
                if (npts==2) {j=i+1; break;}
            }
        }

        if (j>0) {
            if (!strncmp(nom+j,"ffn_up.weight",13) || !strncmp(nom+j,"ffn_gate.weight",15)) {
/*
           ffn_up: row-major: [ab]=d1 1536, +144(avec blk size)   [ac]=d2 8960, +864=row size

           a───b
           │   │
           │   │
           c───┤
           └───┘
		   sizes 1536 8960 7741440 7741440 = no padding !
*/

                int d1 = t->ne[0]; int d2 = t->ne[1];
                int dd1 = t->nb[0]; int dd2 = t->nb[1]; int dd3 = t->nb[2]; int dd4 = t->nb[3];
				size_t rowsz = ggml_row_size(t->type,d1);
				printf("sizes %d %d %d %d %d\n", d1, d2, ggml_nbytes(t), ggml_nbytes_pad(t), rowsz);
				printf("extend %s %d %d %d %d\n",nom, dd1, dd2, dd3, dd4);
				struct ggml_init_params params = {
					/*.mem_size   =*/ rowsz*(d2+1)+ggml_tensor_overhead(),
					/*.mem_buffer =*/ NULL,
					/*.no_alloc   =*/ false,
				};
				struct ggml_context * detctx = ggml_init(params);
				struct ggml_tensor * tt = ggml_new_tensor_2d(detctx, t->type, d1, d2+1);
				ggml_set_name(tt,t->name);
				t = tt;
				printf("alloc OK %d %d\n",tt->ne[0], tt->ne[1]);

                char *new_data = (char *)ggml_get_data(tt);
				memcpy(new_data, buf, rowsz*d2);
				printf("copy OK\n");
                // Initialize new row to zero
                for (int i = d2; i < d2+1; ++i) {
                    memset(new_data + i * rowsz, 0, rowsz);
                }
				printf("new row OK %d\n", n_bytes);

				// write tensor data + padding
				n_bytes = ggml_nbytes(tt);
				fout.write(new_data, n_bytes);
				zeros(fout, GGML_PAD(n_bytes, GGUF_DEFAULT_ALIGNMENT) - n_bytes);
				printf("metadat OK %d\n", n_bytes);
            } else if (false && !strncmp(nom+j,"ffn_down.weight",15)) {
/*
           ffn_down: row-major: [ab]=d1 8960 [ac]=d2 1536

           a─────b┐
           │     ││
           c─────┴┘
*/
                int d1 = t->ne[0]; int d2 = t->ne[1];
                int dd1 = t->nb[0]; int dd2 = t->nb[1]; int dd3 = t->nb[2]; int dd4 = t->nb[3];
				size_t rowsz = ggml_row_size(t->type,d1);
				printf("sizes %d %d %d %d %d\n", d1, d2, ggml_nbytes(t), ggml_nbytes_pad(t), rowsz);
				printf("extend %s %d %d %d %d\n",nom, dd1, dd2, dd3, dd4);
				struct ggml_init_params params = {
					/*.mem_size   =*/ (rowsz+1)*d2+ggml_tensor_overhead(),
					/*.mem_buffer =*/ NULL,
					/*.no_alloc   =*/ false,
				};
				struct ggml_context * detctx = ggml_init(params);
				struct ggml_tensor * tt = ggml_new_tensor_2d(detctx, t->type, d1, d2);
				ggml_set_name(tt,t->name);
				t= tt;
				printf("alloc OK\n");

                char *new_data = (char *)ggml_get_data(tt);
				memcpy(new_data, buf, rowsz*d2);
				printf("copy OK\n");
                // Initialize new row to zero
                for (int i = d2; i < d2+1; ++i) {
                    memset(new_data + i * rowsz, 0, rowsz);
                }
				printf("new row OK\n");

				// write tensor data + padding
				n_bytes = ggml_nbytes(tt);
				fout.write(new_data, n_bytes);
				zeros(fout, GGML_PAD(n_bytes, GGUF_DEFAULT_ALIGNMENT) - n_bytes); 

                // TODO
            } else {
				// write tensor data + padding
				fout.write(buf, n_bytes);
				zeros(fout, GGML_PAD(n_bytes, GGUF_DEFAULT_ALIGNMENT) - n_bytes);
			}
        } else {
			// write tensor data + padding
			fout.write(buf, n_bytes);
			zeros(fout, GGML_PAD(n_bytes, GGUF_DEFAULT_ALIGNMENT) - n_bytes);
		}
		printf("add tensor %s\n",t->name);
		gguf_add_tensor(ctx_out, t);
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
    copy();
    print();
}
