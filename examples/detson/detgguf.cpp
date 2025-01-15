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

void print(int layer_to_modify) {
    struct ggml_context * ctx_meta = NULL;
    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &ctx_meta,
    };
    std::vector<uint8_t> read_data;

    const char * model_path = ("rec_" + std::to_string(layer_to_modify) + ".gguf").c_str();
    auto * ctx_gguf = gguf_init_from_file(model_path, params);

    auto n_tensors = gguf_get_n_tensors(ctx_gguf);
	printf("ntensors %d\n",n_tensors);
    for (int i_tensor = 0; i_tensor < n_tensors; i_tensor++) {
        const char * t_name = gguf_get_tensor_name(ctx_gguf, i_tensor);
        struct ggml_tensor * t = ggml_get_tensor(ctx_meta, t_name);
        printf("tensor %s %d %d\n", t->name, t->ne[0], t->ne[1]);
    }
}

void copy(const char * model_path, int layer_to_modify) {
    FILE *acts_file = fopen("acts.bin.gld","rb");
    FILE *norm_file = fopen("norm.bin.err","rb");
	int vec_dim=0;
	fread(&vec_dim, sizeof(int), 1, acts_file);
    fread(&vec_dim, sizeof(int), 1, norm_file);
    float gld_acts[vec_dim];
    float err_norm[vec_dim];
    int layer = -1;

    // TODO add error handling
    while (layer != layer_to_modify) {
        fread(&layer, sizeof(int), 1, acts_file);
        fread(&layer, sizeof(int), 1, norm_file);

        fread(gld_acts, sizeof(float), vec_dim, acts_file);
        fread(err_norm, sizeof(float), vec_dim, norm_file);
    }

    
    struct ggml_context * ctx_meta = NULL;
    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &ctx_meta,
    };
	char *buf = (char *)malloc(500000000);

    // struct gguf_context * ctx_gguf;
    auto * ctx_gguf = gguf_init_from_file(model_path, params);
    auto * ctx_out = gguf_init_empty();

    std::ofstream fout("rec_" + std::to_string(layer_to_modify) + ".gguf", std::ios::binary);
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
    std::ifstream f_input(model_path, std::ios::binary);
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

				struct ggml_init_params params = {
					/*.mem_size   =*/ (d1*d2+d1*(d2+256))*ggml_type_size(GGML_TYPE_F32) 
						+ (d1*(d2+256))*ggml_type_size(t->type)
						+ 3*ggml_tensor_overhead(),
					/*.mem_buffer =*/ NULL,
					/*.no_alloc   =*/ false,
				};
				struct ggml_context * detctx = ggml_init(params);

				{
					float *bufF32 = (float *)malloc(d1*(d2+256)*sizeof(float));
					// 1- convertit tensor en F32
					printf("dequantize %s\n",ggml_type_name(t->type));
					const auto * qtype = ggml_get_type_traits(t->type); 
					// buf contient les data quantized
					size_t rowsz = ggml_row_size(t->type,d1);
					for (int i=0;i<d2;i++) {
						qtype->to_float(buf+i*rowsz, &bufF32[i*d1], d1); 
                    }
                    for (int nvec=0; nvec<256; nvec++) {
                            for (int i=0; i<d1; i++) {
                            // 3- set values in new column
                            if (atoi(nom+4) == layer_to_modify && nvec==0){
                                // bufF32[d1*(d2+nvec)+i]=err_norm[i];
                                bufF32[d1*(d2+nvec)+i]=0.0;
                                // bufF32[d1*(d2+nvec)+i]=bufF32[d1*(d2-nvec-1)+i];
                            }
                            else {
                                bufF32[d1*(d2+nvec)+i]=0.0;
                            }
                        }
                    }
					printf("dequant done\n");

					// 4- requantize tensor
					struct ggml_tensor * tt = ggml_new_tensor_2d(detctx, t->type, d1, d2+256);
					ggml_set_name(tt,t->name);
					ggml_quantize_chunk(t->type, bufF32, tt->data, 0, d2+256, d1, NULL);
					n_bytes = ggml_nbytes(tt);
					t = tt;
					free(bufF32);
				}
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
					/*.mem_size   =*/ (d1*d2+(d1+256)*d2)*ggml_type_size(GGML_TYPE_F32) 
						+ ((d1+256)*d2)*ggml_type_size(t->type)
						+ 3*ggml_tensor_overhead(),
					/*.mem_buffer =*/ NULL,
					/*.no_alloc   =*/ false,
				};
				struct ggml_context * detctx = ggml_init(params);

				{
					float *bufF32 = (float *)malloc((d1+256)*d2*sizeof(float));
					// 1- convertit tensor en F32
					printf("dequantize %s\n",ggml_type_name(t->type));
					const auto * qtype = ggml_get_type_traits(t->type); 
					// buf contient les data quantized
					size_t rowsz = ggml_row_size(t->type,d1);
					for (int i=0;i<d2;i++) {
						qtype->to_float(buf+i*rowsz, &bufF32[i*(d1+256)], d1); 
						// 3- set values in new column
                        for (int nvec=0; nvec<256; nvec++) {
                            if (atoi(nom+4) == layer_to_modify && nvec==0){
                                // bufF32[(i+1)*d1+i*256+nvec]=gld_acts[i];
                                bufF32[(i+1)*d1+i*256+nvec]=0.0;
                                // bufF32[(i+1)*d1+i*256+nvec]=bufF32[(i+1)*d1+i*256-nvec-1];
                            }
                            else {
                                bufF32[(i+1)*d1+i*256+nvec]=0.0;
                            }
                        }
					}
					printf("dequant done\n");

					// 4- requantize tensor
					struct ggml_tensor * tt = ggml_new_tensor_2d(detctx, t->type, d1+256, d2);
					ggml_set_name(tt,t->name);
					ggml_quantize_chunk(t->type, bufF32, tt->data, 0, d2, d1+256, NULL);
					n_bytes = ggml_nbytes(tt);
					t = tt;
					free(bufF32);
				}
            }
        } 
        // write tensor data + padding
        fout.write(buf, n_bytes);
        zeros(fout, GGML_PAD(n_bytes, GGUF_DEFAULT_ALIGNMENT) - n_bytes);
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
		for (int i=0;i<data.size();i++) {
			unsigned char *c=&data.data()[i];
			if (!strncmp((const char *)c,"feed_forward_length",19)) {
				unsigned char *buffer = (unsigned char *)&data.data()[i+23];
				// int num = (int)buffer[3] | (int)buffer[2]<<8 | (int)buffer[1]<<16 | (int)buffer[0]<<24;
				int num = (int)buffer[0] | (int)buffer[1]<<8 | (int)buffer[2]<<16 | (int)buffer[3]<<24;
				printf("%s %d\n",c,num);
				int *cc = (int *)buffer;
				cc[0] = num+256;
				break;
			}
		}
        fout.write((const char *)data.data(), data.size());
        fout.close();
        gguf_free(ctx_out);
    }
}

int main(int argc, const char ** argv) {
    for (int i=10; i<13; i++) {
        copy(argv[1], i);
        print(i);
    }
}
