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

char MOD[1000] = "/home/xtof/nvme/qwen2/qwen2.5-0.5b-instruct-q5_k_m.gguf";
int ADDDIM = 256;

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
    // auto * ctx_gguf = gguf_init_from_file("tmp.gguf", params);
    auto * ctx_gguf = gguf_init_from_file(MOD, params);

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
    auto * ctx_gguf = gguf_init_from_file(MOD, params);
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
    std::ifstream f_input(MOD, std::ios::binary);
    if (!f_input.is_open()) {
        fprintf(stderr, "%s:  failed to open input GGUF \n", __func__);
        gguf_free(ctx_gguf);
        gguf_free(ctx_out);
        fout.close();
        exit(EXIT_FAILURE);
    }

	gguf_free(ctx_out);
    ctx_out = gguf_init_empty();
    gguf_set_kv(ctx_out, ctx_gguf);
 
	int olddim = -1;
    for (int i_tensor = 0; i_tensor < n_tensors; i_tensor++) {
        const char * t_name = gguf_get_tensor_name(ctx_gguf, i_tensor);
        struct ggml_tensor * t = ggml_get_tensor(ctx_meta, t_name);
		printf("add tensor %s\n",t_name);

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
				olddim = d2;
                int dd1 = t->nb[0]; int dd2 = t->nb[1]; int dd3 = t->nb[2]; int dd4 = t->nb[3];
				size_t rowsz = ggml_row_size(t->type,d1);
				printf("sizes %d %d %d %d %d\n", d1, d2, ggml_nbytes(t), ggml_nbytes_pad(t), rowsz);
				printf("extend %s %d %d %d %d\n",nom, dd1, dd2, dd3, dd4);
				struct ggml_init_params params = {
					/*.mem_size   =*/ rowsz*(d2+ADDDIM)+ggml_tensor_overhead(),
					/*.mem_buffer =*/ NULL,
					/*.no_alloc   =*/ false,
				};
				struct ggml_context * detctx = ggml_init(params);
				struct ggml_tensor * tt = ggml_new_tensor_2d(detctx, t->type, d1, d2+ADDDIM);
				ggml_set_name(tt,t->name);
                gguf_add_tensor(ctx_out, tt);
				printf("alloc OK %d %d\n",tt->ne[0], tt->ne[1]);

                char *new_data = (char *)ggml_get_data(tt);
				memcpy(new_data, buf, rowsz*d2);

				printf("copy OK\n");
                if (ADDDIM>0)
                    // Initialize new row to zero
                    for (int i = d2; i < d2+ADDDIM; ++i) {
                        memset(new_data + i * rowsz, 0.0, rowsz);
                    }
				printf("new row OK %d\n", n_bytes);

				// write tensor data + padding
				n_bytes = ggml_nbytes(tt);
				fout.write(new_data, n_bytes);
                size_t pad = GGML_PAD(n_bytes, gguf_get_alignment(ctx_out)) - n_bytes;
                for (size_t j = 0; j < pad; ++j) {
                    fout.put(0);
                }
                gguf_set_tensor_type(ctx_out, tt->name, t->type);
                gguf_set_tensor_data(ctx_out, tt->name, tt->data, n_bytes);
				// zeros(fout, GGML_PAD(n_bytes, GGUF_DEFAULT_ALIGNMENT) - n_bytes);
				printf("metadat OK %d\n", n_bytes);
            } else if (!strncmp(nom+j,"ffn_down.weight",15)) {
/*
           ffn_down: row-major: [ab]=d1 8960 [ac]=d2 1536

           a─────b┐
           │     ││
           c─────┴┘
*/
                int d1 = t->ne[0]; int d2 = t->ne[1];
				int dd1 = t->nb[0]; int dd2 = t->nb[1]; int dd3 = t->nb[2]; int dd4 = t->nb[3];

				struct ggml_init_params params = {
					/*.mem_size   =*/ (d1*d2+(d1+ADDDIM)*d2)*ggml_type_size(GGML_TYPE_F32) 
						+ ((d1+ADDDIM)*d2)*ggml_type_size(t->type)
						+ 3*ggml_tensor_overhead(),
					/*.mem_buffer =*/ NULL,
					/*.no_alloc   =*/ false,
				};
				struct ggml_context * detctx = ggml_init(params);

				{
					float *bufF32 = (float *)malloc((d1+ADDDIM)*d2*sizeof(float));
					// 1- convertit tensor en F32
                    if (ggml_quantize_requires_imatrix(t->type)) {
                        printf("ERRRRRRRRRR quantize imatrix\n");
                        exit(1);
                    }
					printf("dequantize %s\n",ggml_type_name(t->type));
					const auto * qtype = ggml_get_type_traits(t->type); 
					// buf contient les data quantized
					size_t rowsz = ggml_row_size(t->type,d1);
					for (int i=0;i<d2;i++) {
						qtype->to_float(buf+i*rowsz, &bufF32[i*(d1+ADDDIM)], d1); 
						// 3- set values in new column
                        if (ADDDIM>0) 
                            for (int ii=0;ii<ADDDIM;ii++) bufF32[(i+1)*(d1+ADDDIM)-ii-1]=0.;
					}
					printf("dequant done\n");

					// 4- requantize tensor
					struct ggml_tensor * tt = ggml_new_tensor_2d(detctx, t->type, d1+ADDDIM, d2);
					ggml_set_name(tt,t->name);
                    gguf_add_tensor(ctx_out, tt);
					n_bytes = ggml_quantize_chunk(t->type, bufF32, tt->data, 0, d2, d1+ADDDIM, NULL);
                    gguf_set_tensor_type(ctx_out, tt->name, t->type);
                    gguf_set_tensor_data(ctx_out, tt->name, tt->data, n_bytes);
					t = tt;
					free(bufF32);
				}

				// write tensor data + padding
				fout.write(buf, n_bytes);
                size_t pad = GGML_PAD(n_bytes, gguf_get_alignment(ctx_out)) - n_bytes;
                for (size_t j = 0; j < pad; ++j) {
                    fout.put(0);
                }
				// zeros(fout, GGML_PAD(n_bytes, GGUF_DEFAULT_ALIGNMENT) - n_bytes); 
            } else {
				// write tensor data + padding
				// buf contient les data chargees depuis le fichier d'origine, t n'as pas change
                gguf_add_tensor(ctx_out, t);
				fout.write(buf, n_bytes);
                size_t pad = GGML_PAD(n_bytes, gguf_get_alignment(ctx_out)) - n_bytes;
                for (size_t j = 0; j < pad; ++j) {
                    fout.put(0);
                }
				// zeros(fout, GGML_PAD(n_bytes, GGUF_DEFAULT_ALIGNMENT) - n_bytes);
			}
        } else {
			// write tensor data + padding
            gguf_add_tensor(ctx_out, t);
			fout.write(buf, n_bytes);
            size_t pad = GGML_PAD(n_bytes, gguf_get_alignment(ctx_out)) - n_bytes;
            for (size_t j = 0; j < pad; ++j) {
                fout.put(0);
            }
			// zeros(fout, GGML_PAD(n_bytes, GGUF_DEFAULT_ALIGNMENT) - n_bytes);
		}
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
		for (int i=0;i<data.size();i++) {
			unsigned char *c=&data.data()[i];
			if (ADDDIM>0 && !strncmp((const char *)c,"feed_forward_length",19)) {
				unsigned char *buffer = (unsigned char *)&data.data()[i+23];
				// int num = (int)buffer[3] | (int)buffer[2]<<8 | (int)buffer[1]<<16 | (int)buffer[0]<<24;
				int num = (int)buffer[0] | (int)buffer[1]<<8 | (int)buffer[2]<<16 | (int)buffer[3]<<24;
				printf("%s %d\n",c,num);
				int *cc = (int *)buffer;
				cc[0] = num+ADDDIM;
				break;
			}
		}
        fout.write((const char *)data.data(), data.size());
        fout.close();

    // gguf_write_to_file(ctx_out, "tmp.gguf", false);


        gguf_free(ctx_out);
    }
}

int main(int argc, const char ** argv) {
    copy();
    print();
}
