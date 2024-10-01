#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"

#include <cstdio>
#include <string>
#include <cstring>
#include <vector>

/**
 * This the arbitrary data which will be passed to each callback.
 * Later on we can for example add operation or tensor name filter from the CLI arg, or a file descriptor to dump the tensor.
 */
struct callback_data {
    std::vector<uint8_t> data;
};

static std::string ggml_ne_string(const ggml_tensor * t) {
    std::string str;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        str += std::to_string(t->ne[i]);
        if (i + 1 < GGML_MAX_DIMS) {
            str += ", ";
        }
    }
    return str;
}

static void detson_save_tensor(uint8_t * data, ggml_type type, const int64_t * ne, const size_t * nb, int curlayer) {
    float sum = 0;
    for (int64_t i3 = 0; i3 < ne[3]; i3++) {
        for (int64_t i2 = 0; i2 < ne[2]; i2++) {
            for (int64_t i1 = 0; i1 < ne[1]; i1++) {
                for (int64_t i0 = 0; i0 < ne[0]; i0++) {
                    size_t i = i3 * nb[3] + i2 * nb[2] + i1 * nb[1] + i0 * nb[0];
                    float v;
                    if (type == GGML_TYPE_F16) {
                        v = ggml_fp16_to_fp32(*(ggml_fp16_t *) &data[i]);
                    } else if (type == GGML_TYPE_F32) {
                        v = *(float *) &data[i];
                    } else if (type == GGML_TYPE_I32) {
                        v = (float) *(int32_t *) &data[i];
                    } else if (type == GGML_TYPE_I16) {
                        v = (float) *(int16_t *) &data[i];
                    } else if (type == GGML_TYPE_I8) {
                        v = (float) *(int8_t *) &data[i];
                    } else {
                        GGML_ABORT("fatal error");
                    }
                    if (sum==0) {
                        // just print the first value
                        printf("detson %d %f\n",curlayer,v);
                    }
                    printf("DETAC %d %d %d %d %d %f\n",curlayer,i0,i1,i2,i3,v);
                    sum += v;
                }
            }
        }
    }
}

static void ggml_print_tensor(uint8_t * data, ggml_type type, const int64_t * ne, const size_t * nb, int64_t n) {
    GGML_ASSERT(n > 0);
    float sum = 0;
    for (int64_t i3 = 0; i3 < ne[3]; i3++) {
        LOG("                                     [\n");
        for (int64_t i2 = 0; i2 < ne[2]; i2++) {
            if (i2 == n && ne[2] > 2*n) {
                LOG("                                      ..., \n");
                i2 = ne[2] - n;
            }
            LOG("                                      [\n");
            for (int64_t i1 = 0; i1 < ne[1]; i1++) {
                if (i1 == n && ne[1] > 2*n) {
                    LOG("                                       ..., \n");
                    i1 = ne[1] - n;
                }
                LOG("                                       [");
                for (int64_t i0 = 0; i0 < ne[0]; i0++) {
                    if (i0 == n && ne[0] > 2*n) {
                        LOG("..., ");
                        i0 = ne[0] - n;
                    }
                    size_t i = i3 * nb[3] + i2 * nb[2] + i1 * nb[1] + i0 * nb[0];
                    float v;
                    if (type == GGML_TYPE_F16) {
                        v = ggml_fp16_to_fp32(*(ggml_fp16_t *) &data[i]);
                    } else if (type == GGML_TYPE_F32) {
                        v = *(float *) &data[i];
                    } else if (type == GGML_TYPE_I32) {
                        v = (float) *(int32_t *) &data[i];
                    } else if (type == GGML_TYPE_I16) {
                        v = (float) *(int16_t *) &data[i];
                    } else if (type == GGML_TYPE_I8) {
                        v = (float) *(int8_t *) &data[i];
                    } else {
                        GGML_ABORT("fatal error");
                    }
                    LOG("%12.4f", v);
                    sum += v;
                    if (i0 < ne[0] - 1) LOG(", ");
                }
                LOG("],\n");
            }
            LOG("                                      ],\n");
        }
        LOG("                                     ]\n");
        LOG("                                     sum = %f\n", sum);
    }
}

/**
 * GGML operations callback during the graph execution.
 *
 * @param t current tensor
 * @param ask when ask is true, the scheduler wants to know if we are interested in data from this tensor
 *            if we return true, a follow-up call will be made with ask=false in which we can do the actual collection.
 *            see ggml_backend_sched_eval_callback
 * @param user_data user data to pass at each call back
 * @return true to receive data or continue the graph, false otherwise
 */
static bool ggml_debug(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * cb_data = (callback_data *) user_data;

    const struct ggml_tensor * src0 = t->src[0];
    const struct ggml_tensor * src1 = t->src[1];

    if (ask) {
        return true; // Always retrieve data
    }

    char src1_str[128] = {0};
    if (src1) {
        snprintf(src1_str, sizeof(src1_str), "%s{%s}", src1->name, ggml_ne_string(src1).c_str());
    }

    /*
    LOG("%s: %24s = (%s) %10s(%s{%s}, %s}) = {%s}\n", __func__,
         t->name, ggml_type_name(t->type), ggml_op_desc(t),
         src0->name, ggml_ne_string(src0).c_str(),
         src1 ? src1_str : "",
         ggml_ne_string(t).c_str());
         */

    // copy the data from the GPU memory if needed
    const bool is_host = ggml_backend_buffer_is_host(t->buffer);

    if (!is_host) {
        auto n_bytes = ggml_nbytes(t);
        cb_data->data.resize(n_bytes);
        ggml_backend_tensor_get(t, cb_data->data.data(), 0, n_bytes);
    }
    
    printf("detsonlayer %s %s %d %d %d %d\n",t->name, ggml_op_desc(t), t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
    if (!strncmp(t->name,"result_output",13)) {
        // tt = full embedding matrix
        struct ggml_tensor *tt = t->src[0];
        printf("detsonlayerprev %s %s %d %d %d %d\n",tt->name, ggml_op_desc(tt), tt->ne[0], tt->ne[1], tt->ne[2], tt->ne[3]);
        const bool is_host = ggml_backend_buffer_is_host(tt->buffer);
        if (!is_host) {
            auto n_bytes = ggml_nbytes(tt);
            cb_data->data.resize(n_bytes);
            ggml_backend_tensor_get(tt, cb_data->data.data(), 0, n_bytes);
        }
        uint8_t * data = is_host ? (uint8_t *) tt->data : cb_data->data.data();

        if (tt->type != GGML_TYPE_F32) {
            auto nels = ggml_nelements(tt);
            ggml_type_traits_t qtype = ggml_internal_get_type_traits(tt->type);
            // TODO: bug ? dequantized are not reasonable floats...
            std::vector<uint8_t> dequant_buf(nels * sizeof(float));
            qtype.to_float(data, (float *)dequant_buf.data(), nels);
            float *dqbuf = (float *)dequant_buf.data();
            printf("detsondbug %d %f %f %f\n",nels, dqbuf[100], dqbuf[101], dqbuf[102]);
            FILE *f = fopen("embs.bin","wb");
            fwrite(dqbuf,sizeof(float),nels,f);
            fclose(f);
        }
    }
    if (!ggml_is_quantized(t->type)) {
        if (!strncmp(t->name,"l_out",5)) {
            printf("detson save %s\n",t->name);
            int curlay = atoi(t->name+6);
            uint8_t * data = is_host ? (uint8_t *) t->data : cb_data->data.data();
            detson_save_tensor(data, t->type, t->ne, t->nb, curlay);
        }
    }

    return true;
}

static bool run(llama_context * ctx, const gpt_params & params) {
    const bool add_bos = llama_add_bos_token(llama_get_model(ctx));

    std::vector<llama_token> tokens = ::llama_tokenize(ctx, params.prompt, add_bos);
    for (int i=0;i<tokens.size();i++) printf("dettokens %d %d\n",i,tokens[i]);

    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size(), 0, 0))) {
        LOG_ERR("%s : failed to eval\n", __func__);
        return false;
    }

    return true;
}

int main(int argc, char ** argv) {
    callback_data cb_data;

    gpt_params params;

    if (!gpt_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    gpt_init();

    llama_backend_init();
    llama_numa_init(params.numa);

    // pass the callback to the backend scheduler
    // it will be executed for each node during the graph computation
    params.cb_eval = ggml_debug;
    params.cb_eval_user_data = &cb_data;
    params.warmup = false;

    // init
    llama_init_result llama_init = llama_init_from_gpt_params(params);

    llama_model * model = llama_init.model;
    llama_context * ctx = llama_init.context;
    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s : failed to init\n", __func__);
        return 1;
    }

    // print system information
    {
        LOG_INF("\n");
        LOG_INF("%s\n", gpt_params_get_system_info(params).c_str());
        LOG_INF("\n");
    }

    bool OK = run(ctx, params);
    if (!OK) {
        return 1;
    }

    LOG("\n");
    llama_perf_context_print(ctx);

    llama_free(ctx);
    llama_free_model(model);

    llama_backend_free();

    return 0;
}
