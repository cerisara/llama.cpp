#include "server-context.h"
#include "server-http.h"
#include "server-models.h"
#include "server-cors-proxy.h"
#include "server-stream.h"
#include "server-tools.h"

#include "arg.h"
#include "build-info.h"
#include "common.h"
#include "fit.h"
#include "llama.h"
#include "log.h"

#include <atomic>
#include <clocale>
#include <exception>
#include <signal.h>
#include <thread> // for std::thread::hardware_concurrency

#if defined(_WIN32)
#include <windows.h>
#endif

// detson semaphore to communicate with python
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <semaphore.h>
#include <unistd.h>
#include <iostream>
#include <cstring>
#include <vector>
#include <cinttypes>
#include <stdlib.h>

static const char* SHM_NAME = "/ring_buffer_demo";
static const char* SEM_C2P = "/c2py_sem";
static const char* SEM_P2C = "/py2c_sem";
struct SharedMemory {
    // big enough for one full-prompt hidden-state tensor: ubatch (16384) x hidden (4096)
    float buffers[1][70000000];
};
SharedMemory *shm;
sem_t* sem_c2p;
sem_t* sem_py2c;

char **detsavelayer = (char **)malloc(sizeof(char *)*1000);
char *detsaveemb = NULL; // when set (from SAVE_EMB), save embeddings then stop
int detembsaved = 0; // once saved, do not save it again
char * firstnodeseen = NULL; // first node name seen by the eval callback
int done_first_node = 0; // stop printing node names once the sequence is known
struct callback_data {
    std::vector<uint8_t> data;
};

static void detson_send_tensor(const char * name, uint8_t * data, ggml_type type, const int64_t * ne, const size_t * nb) {
    float sum = 0;

    // Fill buffer
    int bufidx = 0;
    char namebuf[100];
    strncpy(namebuf, name, 99);
    namebuf[99] = '\0';
    for (int i = 0; i < 100; i++) {
        shm->buffers[0][bufidx++] = (float)(unsigned char)namebuf[i];
    }
    for (int64_t i3 = 0; i3 < ne[3]; i3++) {
        for (int64_t i2 = 0; i2 < ne[2]; i2++) {
            int32_t val = ne[1];
            if (i3==0 && i2==0) {
                shm->buffers[0][bufidx++] = float(val);
            }
            for (int64_t i1 = 0; i1 < ne[1]; i1++) {
                int32_t val2 = ne[0];
                if (i3==0 && i2==0 && i1==0) {
                    // printf("detne %d %d\n",val,val2);
                    shm->buffers[0][bufidx++] = float(val2);
                }
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
                        printf("[detson_send_tensor] error: unsupported tensor type %s\n", ggml_type_name(type));
                        v = 0.0f;
                    }
                    if (i0==0) {
                        // pour check en python que vector parse dans les bonnes dim
                        // printf("vec %d %f\n",i1,v);
                    }
                    shm->buffers[0][bufidx++] = v;
                    sum += v;
                }
            }
        }
    }
    // std::cout << "[C++] Sending buffer " << bufidx << " " << sum << "\n";
    // Notify Python
    sem_post(sem_c2p);
    int r = sem_wait(sem_py2c);
}

static bool detsoncb_share_activs(struct ggml_tensor * t, bool ask, void * user_data) {
    if (ask) return true; // Always retrieve data
    auto * cb_data = (callback_data *) user_data;
    const struct ggml_tensor * src0 = t->src[0];
    const struct ggml_tensor * src1 = t->src[1];

	// Debug block to inspect inputs and output of the final matrix multiplication
	/* WARNING: this creates a crash with MoE !!!!!
	   Why does it read OOB specifically on MoE? With --n-cpu-moe 30 the graph
	   is split across GPU/CPU, and the last-layer tensors the eval callback
	   sees are partial/split views; the debug block's
	   ggml_backend_tensor_get(...) + offset math ((ne[1]-1)*ne[0]*...) overruns
	   the tensor's real allocation. Dense with -ngl 99 has no such splits, so
	   it doesn't crash.
	   */
	/*
	if (t->op == GGML_OP_MUL_MAT && src0 != NULL && strncmp(src0->name, "output.weight", 13) == 0) {
        printf("\n--- [DEBUG] Logits Calculation Detected ---\n");

        // Helper lambda to print the first few values of a tensor
        auto print_tensor_head = [](const struct ggml_tensor * tensor, const char* name, bool last_token_only) {
            if (tensor == NULL) {
                printf("[DEBUG] %s is NULL\n", name);
                return;
            }
            printf("[DEBUG] Head of %s (name: %s, type: %s)\n", name, tensor->name, ggml_type_name(tensor->type));

            const bool is_host = ggml_backend_buffer_is_host(tensor->buffer);
            std::vector<uint8_t> data_host_vec;
            const uint8_t * data_ptr;
            if (!is_host) {
                auto n_bytes = ggml_nbytes(tensor);
                data_host_vec.resize(n_bytes);
                ggml_backend_tensor_get(tensor, data_host_vec.data(), 0, n_bytes);
                data_ptr = data_host_vec.data();
            } else {
                data_ptr = (const uint8_t *) tensor->data;
            }

            const void* data_to_print = data_ptr;
            if (last_token_only && tensor->ne[1] > 0) {
                int64_t n_tokens = tensor->ne[1];
                size_t offset = (n_tokens - 1) * tensor->ne[0] * (ggml_type_size(tensor->type) / ggml_blck_size(tensor->type));
                data_to_print = (const uint8_t*)data_ptr + offset;
            }

            const int n_els_to_print = 10;
            std::vector<float> float_values(n_els_to_print);
            const ggml_type_traits *traits = ggml_get_type_traits(tensor->type);
            if (traits->to_float) {
                traits->to_float(data_to_print, float_values.data(), n_els_to_print);
                printf("[DEBUG] First %d values of %s: ", n_els_to_print, name);
                for(int i = 0; i < n_els_to_print; ++i) {
                    printf("%.6f ", float_values[i]);
                }
                printf("\n");
			} else {
				if (!strncmp(ggml_type_name(tensor->type),"f32",3)) {
					float *float_values = (float *)data_to_print;
					printf("[DEBUG] First %d values of %s: ", n_els_to_print, name);
					for(int i = 0; i < n_els_to_print; ++i) {
						printf("%.6f ", float_values[i]);
					}
					printf("\n"); 
				} else {
					printf("[DEBUG] Cannot print values for type %s\n", ggml_type_name(tensor->type));
				}
			}
        };

        // Print inputs
        print_tensor_head(src1, "Final Activations", true);
        print_tensor_head(src0, "Unembedding Matrix", false);

        // Print output argmax for confirmation
        const bool is_host_out = ggml_backend_buffer_is_host(t->buffer);
        std::vector<uint8_t> data_host_vec_out;
        uint8_t * data_out;
        if (!is_host_out) {
            auto n_bytes = ggml_nbytes(t);
            data_host_vec_out.resize(n_bytes);
            ggml_backend_tensor_get(t, data_host_vec_out.data(), 0, n_bytes);
            data_out = data_host_vec_out.data();
        } else {
            data_out = (uint8_t *) t->data;
        }

        if (t->type == GGML_TYPE_F32) {
            float* logits_data = (float*) data_out;
            int64_t n_tokens = t->ne[1];
            int64_t vocab_size = t->ne[0];
            if (n_tokens > 0) {
                float* last_token_logits = logits_data + (n_tokens - 1) * vocab_size;
                int max_idx = 0;
                for (int i = 1; i < vocab_size; ++i) {
                    if (last_token_logits[i] > last_token_logits[max_idx]) max_idx = i;
                }
                printf("[DEBUG] C++ argmax next token prediction: %d\n", max_idx);
            }
        } else {
            printf("[DEBUG] Output logits tensor is not F32.\n");
        }
        printf("--- [DEBUG] End Logits Calculation ---\n\n");
    }*/

    // copy the data from the GPU memory if needed
    const bool is_host = ggml_backend_buffer_is_host(t->buffer);
    if (!is_host) {
        auto n_bytes = ggml_nbytes(t);
        cb_data->data.resize(n_bytes);
        ggml_backend_tensor_get(t, cb_data->data.data(), 0, n_bytes);
    }

	if (!done_first_node) {
		if (firstnodeseen == NULL) {
			fprintf(stderr,"node: %s\n",t->name);
			firstnodeseen = (char *) t->name;
		} else if (strcmp(firstnodeseen, t->name) == 0) {
			done_first_node = 1;
		} else {
			fprintf(stderr,"node: %s\n",t->name);
		}
	}
    for (int i=0;i<1000;i++) {
        if (detsavelayer[i]==NULL) break;

        if (strlen(detsavelayer[i])==strlen(t->name) && !strncmp(t->name,detsavelayer[i],strlen(detsavelayer[i]))) {
			fprintf(stderr,"detsoncpp detected layer2send\n");
                uint8_t * data = is_host ? (uint8_t *) t->data : cb_data->data.data();
                detson_send_tensor(t->name, data, t->type, t->ne, t->nb);

                if (detsavelayer[i+1]==NULL) {
                    if (shm->buffers[0][0] != 424242.0f) {
						fprintf(stderr,"detsoncpp ecrase last activs\n");
                        // recopie la shared RAM dans le computation graph de llamacpp
                        int bufidx = 102; // skip the 2 first ints = dims and the node name !
                        uint8_t * data = is_host ? (uint8_t *) t->data : cb_data->data.data();
                        for (int64_t i3 = 0; i3 < t->ne[3]; i3++) {
                            for (int64_t i2 = 0; i2 < t->ne[2]; i2++) {
                                for (int64_t i1 = 0; i1 < t->ne[1]; i1++) {
                                    for (int64_t i0 = 0; i0 < t->ne[0]; i0++) {
                                        size_t i = i3 * t->nb[3] + i2 * t->nb[2] + i1 * t->nb[1] + i0 * t->nb[0];
                                        float py_val = shm->buffers[0][bufidx++];
                                        if (t->type == GGML_TYPE_F16) {
                                            ggml_fp16_t *v = (ggml_fp16_t *) &data[i];
                                            *v = ggml_fp32_to_fp16(py_val);
                                        } else if (t->type == GGML_TYPE_F32) {
                                            float *v = (float *) &data[i];
                                            *v = py_val;
                                        }
                                    }
                                }
                            }
                        }
						if (!is_host) {
							ggml_backend_tensor_set(t, cb_data->data.data(), 0, cb_data->data.size());
						}
						fprintf(stderr,"detsoncpp fini ecrase last activs\n");
					}
                }
            }
    }
    return true;
}

void llama_server_terminate();

static bool detsoncb_save_embeds(struct ggml_tensor * t, bool ask, void * user_data) {
    if (ask) return true; // Always retrieve data
	if (detembsaved==1) return detsoncb_share_activs(t, ask, user_data);
    const struct ggml_tensor * src0 = t->src[0];
    const struct ggml_tensor * src1 = t->src[1];

	/*
	fprintf(stderr,"DETSAVENODE %lld %lld %lld %lld\n",
            src0!=NULL ? (long long) src0->ne[0] : -1,
            src0!=NULL ? (long long) src0->ne[1] : -1,
            src1!=NULL ? (long long) src1->ne[0] : -1,
            src1!=NULL ? (long long) src1->ne[1] : -1);
			*/
    // detect the unembedding matrix by its dims, no node name needed: only the
    // vocab dim exceeds 100000, and the latent dim matches the activations src1
    if (detsaveemb!=NULL && t->op == GGML_OP_MUL_MAT && src0!=NULL && src1!=NULL &&
        (src0->ne[0] > 100000 || src0->ne[1] > 100000)) {
        auto * cb_data = (callback_data *) user_data;
        uint8_t * data = (uint8_t *) src0->data;
    
        // copy the data from the GPU memory if needed
        const bool is_host = ggml_backend_buffer_is_host(src0->buffer);
        fprintf(stderr,"saving embeddings %d %d\n", is_host, ggml_is_quantized(src0->type));
        if (!is_host) {
            auto n_bytes = ggml_nbytes(src0);
            cb_data->data.resize(n_bytes);
            ggml_backend_tensor_get(src0, cb_data->data.data(), 0, n_bytes);
            fprintf(stderr,"ERROR GPU not implemented yet");
        }       

        // save unembedding matrix
        if (src0->type != GGML_TYPE_F32) {
            auto nels = ggml_nelements(src0);
            fprintf(stderr,"dequantizing... %d %d %d %d %d\n",nels,
                    src0->ne[3],
                    src0->ne[2],
                    src0->ne[1],
                    src0->ne[0]);
            ggml_type_traits qtype = *ggml_get_type_traits(src0->type);
            std::vector<uint8_t> dequant_buf(nels * sizeof(float));
            qtype.to_float(data, (float *)dequant_buf.data(), nels);
            float *dqbuf = (float *)dequant_buf.data();
            FILE *f = fopen("detembeds.dims","w");
			fprintf(f,"%d\n",src0->ne[3]);
            fprintf(f,"%d\n",src0->ne[2]);
            fprintf(f,"%d\n",src0->ne[1]);
            fprintf(f,"%d\n",src0->ne[0]);
            fclose(f);
            f = fopen("detembeds.bin","wb");
            fwrite(dqbuf,sizeof(float),nels,f);
            fclose(f);
            printf("Embeddings saved; you can rerun the program!");
			detembsaved=1;
            llama_server_terminate();
        }
		fprintf(stderr,"DETSAVEEMBED FINI\n");
    }
	return detsoncb_share_activs(t, ask, user_data);
}

static std::function<void(int)> shutdown_handler;
static std::atomic_flag is_terminating = ATOMIC_FLAG_INIT;

static inline void signal_handler(int signal) {
    if (is_terminating.test_and_set()) {
        // in case it hangs, we can force terminate the server by hitting Ctrl+C twice
        // this is for better developer experience, we can remove when the server is stable enough
        fprintf(stderr, "Received second interrupt, terminating immediately.\n");
        exit(1);
    }

    shutdown_handler(signal);
}

// satisfies -Wmissing-declarations (used by llama command)
int llama_server(int argc, char ** argv);

// to be used via CLI (argc / argv are used by router mode only)
int llama_server(common_params & params, int argc, char ** argv);
void llama_server_terminate();
void llama_server_terminate() {
    if (shutdown_handler) {
        shutdown_handler(0);
    }
}


// wrapper function that handles exceptions and logs errors
// this is to make sure handler_t never throws exceptions; instead, it returns an error response
static server_http_context::handler_t ex_wrapper(server_http_context::handler_t func) {
    return [func = std::move(func)](const server_http_req & req) -> server_http_res_ptr {
        std::string message;
        error_type error;
        try {
            return func(req);
        } catch (const std::invalid_argument & e) {
            // treat invalid_argument as invalid request (400)
            error = ERROR_TYPE_INVALID_REQUEST;
            message = e.what();
        } catch (const std::exception & e) {
            // treat other exceptions as server error (500)
            error = ERROR_TYPE_SERVER;
            message = e.what();
        } catch (...) {
            error = ERROR_TYPE_SERVER;
            message = "unknown error";
        }

        auto res = std::make_unique<server_http_res>();
        res->status = 500;
        try {
            json error_data = format_error_response(message, error);
            res->status = json_value(error_data, "code", 500);
            res->data = safe_json_to_str({{ "error", error_data }});
            SRV_WRN("got exception: %s\n", res->data.c_str());
        } catch (const std::exception & e) {
            SRV_ERR("got another exception: %s | while handling exception: %s\n", e.what(), message.c_str());
            res->data = "Internal Server Error";
        }
        return res;
    };
}

int llama_server(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

#ifndef _WIN32
    // Ignore SIGPIPE so the server does not crash if a child (MCP server, tools runtime) exits while we are writing to its stdin
    signal(SIGPIPE, SIG_IGN);
#endif

    // own arguments required by this example
    common_params params;

    common_init();

    // start the stream session manager GC right after common init, before any HTTP route can
    // touch it. lifecycle is symmetric, stop_gc() runs in clean_up() before backend free
    server_stream_session_manager_start();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    return llama_server(params, argc, argv);
}

int llama_server(common_params & params, int argc, char ** argv) {

// ===== detson: shared-memory activation handoff (ported from sharedram branch) =====
    {
        LOG_WRN("detson llama server\n");
        // Create shared memory
        int fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
        ftruncate(fd, sizeof(SharedMemory));
        void* addr = mmap(nullptr, sizeof(SharedMemory), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        shm = reinterpret_cast<SharedMemory*>(addr);
        LOG_WRN("detson sharedmem1\n");
        // Create semaphores
        sem_c2p = sem_open(SEM_C2P, O_CREAT, 0666, 0);
        sem_py2c = sem_open(SEM_P2C, O_CREAT, 0666, 0);
        LOG_WRN("detson semaphores\n");

        LOG_WRN("detson init\n");
        for (int i=0;i<1000;i++) detsavelayer[i]=NULL;
        LOG_WRN("detsavelayer initialized\n");
        callback_data cb_data;
		const char *detsaveemb_env = getenv("SAVE_EMB");
		if (detsaveemb_env!=NULL) detsaveemb=strdup(detsaveemb_env); // SAVE_EMB only enables saving; the unembedding node is found by its dims
		if (detsaveemb==NULL) {
            LOG_WRN("detson will share activations\n");
            params.cb_eval = detsoncb_share_activs;
        } else {
            LOG_WRN("detson will save embeddings only\n");
            params.cb_eval = detsoncb_save_embeds;
        }
        LOG_WRN("detcallback setup\n");
        params.cb_eval_user_data = &cb_data;
        params.warmup = false;
        {
            int j=0;
            char line[10000];
            FILE *f2 = fopen("layers2save","r");
            if (f2!=NULL) {
                while (fgets(line, sizeof(line), f2) != NULL) {
                    LOG_WRN("detcallback %s %d\n",line,strlen(line));
                    line[strlen(line)-1]=0; // -1 because we remove \n
                    if (strlen(line)==0) break;
                    detsavelayer[j]= (char *)malloc(sizeof(char)*strlen(line));
                    LOG_WRN("detallocline\n");
                    strcpy(detsavelayer[j++],line);
                }
                fclose(f2);
            }
        }
        LOG_WRN("detsavelayer loaded\n");
    }

    bool is_run_by_cli = (argv == nullptr);

    common_models_handler models_handler;

    // note: router mode also accepts -hf remote-preset, so we need to check that first
    if (!is_run_by_cli && !params.model.hf_repo.empty()) {
        try {
            models_handler = common_models_handler_init(params, LLAMA_EXAMPLE_SERVER);
            if (common_models_handler_is_preset_repo(models_handler)) {
                // apply the preset and start the server in router mode
                common_models_handler_apply(models_handler, params);
            }
        } catch (const std::exception & e) {
            SRV_ERR("failed to fetch model metadata: %s\n", e.what());
            return 1;
        }
    }

    // router server never loads a model and must not touch the GPU
    const bool is_router_server = params.model.path.empty()
                               && params.model.hf_repo.empty()
                               && params.model.docker_repo.empty();

    // skip device enumeration so the CUDA primary context stays uncreated
    common_params_print_info(params, !is_router_server);

    if (!is_router_server) {
        // validate batch size for embeddings
        // embeddings require all tokens to be processed in a single ubatch
        // see https://github.com/ggml-org/llama.cpp/issues/12836
        if (params.embedding && params.n_batch > params.n_ubatch) {
            SRV_WRN("embeddings enabled with n_batch (%d) > n_ubatch (%d)\n", params.n_batch, params.n_ubatch);
            SRV_WRN("setting n_batch = n_ubatch = %d to avoid assertion failure\n", params.n_ubatch);
            params.n_batch = params.n_ubatch;
        }

        if (params.n_parallel < 0) {
            SRV_TRC("%s", "n_parallel is set to auto, using n_parallel = 4 and kv_unified = true\n");

            params.n_parallel = 4;
            params.kv_unified = true;
        }
    }

    // size the KV pool from --kv-unified-per-slot, unless the user pinned it with -c
    // or with -c 0 for max context
    const bool ctx_pool_auto_sized = params.kv_unified_per_slot > 0 &&
                                     params.n_ctx == 0 &&
                                     (uint32_t) params.fit_params_min_ctx != UINT32_MAX;

    if (ctx_pool_auto_sized) {
        params.n_ctx = params.n_parallel * params.kv_unified_per_slot;
        SRV_INF("--kv-unified-per-slot: sizing KV pool to n_parallel * kv_unified_per_slot = %d * %d = %d\n", params.n_parallel,
                params.kv_unified_per_slot, params.n_ctx);
    }

    // for consistency between server router mode and single-model mode, we set the same model name as alias
    auto model_name = params.model.get_name();
    if (params.model_alias.empty() && !model_name.empty()) {
        params.model_alias.insert(model_name);
    }

    // note: this is guaranteed to out-live ctx_http and tools
    server_mcp mcp_mgr;

    // struct that contains llama context and inference
    server_context ctx_server;

    server_http_context ctx_http;
    if (!ctx_http.init(params)) {
        SRV_ERR("%s", "failed to initialize HTTP server\n");
        return 1;
    }

    //
    // Router
    //

    // register API routes
    server_child child; // only used in non-router mode
    server_routes routes(params, ctx_server);
    server_tools tools;

    std::optional<server_models_routes> models_routes{};
    if (is_router_server) {
        // setup server instances manager
        try {
            models_routes.emplace(params, argc, argv);
        } catch (const std::exception & e) {
            SRV_ERR("failed to initialize router models: %s\n", e.what());
            return 1;
        }

        // proxy handlers
        // note: routes.get_health stays the same
        routes.get_metrics                 = models_routes->proxy_get;
        routes.post_props                  = models_routes->proxy_post;
        routes.post_completions            = models_routes->proxy_post;
        routes.post_completions_oai        = models_routes->proxy_post;
        routes.post_chat_completions       = models_routes->proxy_post;
        routes.post_control                = models_routes->proxy_post;
        routes.post_responses_oai          = models_routes->proxy_post;
        routes.post_transcriptions_oai     = models_routes->proxy_post;
        routes.post_anthropic_messages     = models_routes->proxy_post;
        routes.post_anthropic_count_tokens = models_routes->proxy_post;
        routes.post_infill                 = models_routes->proxy_post;
        routes.post_embeddings             = models_routes->proxy_post;
        routes.post_embeddings_oai         = models_routes->proxy_post;
        routes.post_rerank                 = models_routes->proxy_post;
        routes.post_tokenize               = models_routes->proxy_post;
        routes.post_detokenize             = models_routes->proxy_post;
        routes.post_apply_template         = models_routes->proxy_post;
        routes.post_chat_completions_tok   = models_routes->proxy_post;
        routes.post_responses_tok_oai      = models_routes->proxy_post;
        routes.get_lora_adapters           = models_routes->proxy_get;
        routes.post_lora_adapters          = models_routes->proxy_post;
        routes.get_slots                   = models_routes->proxy_get;
        routes.post_slots                  = models_routes->proxy_post;

        // custom routes for router
        routes.get_props                   = models_routes->get_router_props;
        routes.get_models                  = models_routes->get_router_models;

        ctx_http.post("/models",               ex_wrapper(models_routes->post_router_models));
        ctx_http.post("/models/load",          ex_wrapper(models_routes->post_router_models_load));
        ctx_http.post("/models/unload",        ex_wrapper(models_routes->post_router_models_unload));
        ctx_http.get ("/models/sse",           ex_wrapper(models_routes->get_router_models_sse));
        ctx_http.del ("/models",               ex_wrapper(models_routes->del_router_models));
    }

    ctx_http.get ("/health",                   ex_wrapper(routes.get_health)); // public endpoint (no API key check)
    ctx_http.get ("/v1/health",                ex_wrapper(routes.get_health)); // public endpoint (no API key check)
    ctx_http.get ("/metrics",                  ex_wrapper(routes.get_metrics));
    ctx_http.get ("/props",                    ex_wrapper(routes.get_props));
    ctx_http.post("/props",                    ex_wrapper(routes.post_props));
    ctx_http.get ("/models",                   ex_wrapper(routes.get_models));
    ctx_http.get ("/v1/models",                ex_wrapper(routes.get_models));
    ctx_http.post("/completion",               ex_wrapper(routes.post_completions)); // legacy
    ctx_http.post("/completions",              ex_wrapper(routes.post_completions));
    ctx_http.post("/v1/completions",           ex_wrapper(routes.post_completions_oai));
    ctx_http.post("/chat/completions",         ex_wrapper(routes.post_chat_completions));
    ctx_http.post("/v1/chat/completions",      ex_wrapper(routes.post_chat_completions));
    ctx_http.post("/v1/chat/completions/control", ex_wrapper(routes.post_control));
    ctx_http.post("/v1/responses",             ex_wrapper(routes.post_responses_oai));
    ctx_http.post("/responses",                ex_wrapper(routes.post_responses_oai));
    ctx_http.post("/v1/audio/transcriptions",  ex_wrapper(routes.post_transcriptions_oai));
    ctx_http.post("/audio/transcriptions",     ex_wrapper(routes.post_transcriptions_oai));
    ctx_http.post("/v1/messages",              ex_wrapper(routes.post_anthropic_messages)); // anthropic messages API
    ctx_http.post("/infill",                   ex_wrapper(routes.post_infill));
    ctx_http.post("/embedding",                ex_wrapper(routes.post_embeddings)); // legacy
    ctx_http.post("/embeddings",               ex_wrapper(routes.post_embeddings));
    ctx_http.post("/v1/embeddings",            ex_wrapper(routes.post_embeddings_oai));
    ctx_http.post("/rerank",                   ex_wrapper(routes.post_rerank));
    ctx_http.post("/reranking",                ex_wrapper(routes.post_rerank));
    ctx_http.post("/v1/rerank",                ex_wrapper(routes.post_rerank));
    ctx_http.post("/v1/reranking",             ex_wrapper(routes.post_rerank));
    ctx_http.post("/tokenize",                 ex_wrapper(routes.post_tokenize));
    ctx_http.post("/detokenize",               ex_wrapper(routes.post_detokenize));
    ctx_http.post("/apply-template",           ex_wrapper(routes.post_apply_template));
    // token counting
    ctx_http.post("/chat/completions/input_tokens",    ex_wrapper(routes.post_chat_completions_tok));
    ctx_http.post("/v1/chat/completions/input_tokens", ex_wrapper(routes.post_chat_completions_tok));
    ctx_http.post("/responses/input_tokens",           ex_wrapper(routes.post_responses_tok_oai));
    ctx_http.post("/v1/responses/input_tokens",        ex_wrapper(routes.post_responses_tok_oai));
    ctx_http.post("/v1/messages/count_tokens",         ex_wrapper(routes.post_anthropic_count_tokens)); // anthropic token counting
    // LoRA adapters hotswap
    ctx_http.get ("/lora-adapters",            ex_wrapper(routes.get_lora_adapters));
    ctx_http.post("/lora-adapters",            ex_wrapper(routes.post_lora_adapters));
    // Save & load slots
    ctx_http.get ("/slots",                    ex_wrapper(routes.get_slots));
    ctx_http.post("/slots/:id_slot",           ex_wrapper(routes.post_slots));

    // resumable streaming: a child binds the local session factories, the router binds
    // proxies that resolve the owning child, see server-stream.h
    server_http_context::handler_t stream_get_h;
    server_http_context::handler_t streams_lookup_h;
    server_http_context::handler_t stream_delete_h;
    if (is_router_server) {
        stream_get_h     = models_routes->router_stream_get;
        streams_lookup_h = models_routes->router_streams_lookup;
        stream_delete_h  = models_routes->router_stream_delete;
    } else {
        stream_get_h     = server_stream_make_get_handler();
        streams_lookup_h = server_stream_make_lookup_handler();
        stream_delete_h  = server_stream_make_delete_handler();
    }
    ctx_http.get ("/v1/stream",                ex_wrapper(stream_get_h));
    ctx_http.post("/v1/streams/lookup",        ex_wrapper(streams_lookup_h));
    ctx_http.del ("/v1/stream",                ex_wrapper(stream_delete_h));

    // Google Cloud Platform (Vertex AI) compat
    ctx_http.register_gcp_compat();

    // return 403 for disabled features
    server_http_context::handler_t res_403 = [](const server_http_req &) {
        auto res = std::make_unique<server_http_res>();
        res->status = 403;
        res->data = safe_json_to_str({
            {"error", {
                {"message", "this feature is disabled"},
                {"type", "feature_disabled"},
            }}
        });
        return res;
    };

    if (params.cors_origins == "*" && params.api_keys.empty()) {
        SRV_WRN("%s", "-----------------\n");
        SRV_WRN("%s", "CORS is set to allow all origins ('*') and no API key is set\n");
        SRV_WRN("%s", "this can be a security risk (cross-origin attacks)\n");
        SRV_WRN("%s", "more info: https://github.com/ggml-org/llama.cpp/pull/25655\n");
        SRV_WRN("%s", "-----------------\n");
    }

    // CORS proxy (EXPERIMENTAL, only used by the Web UI for MCP)
    std::vector<std::string> warn_names;
    if (is_router_server) {
        warn_names.push_back("router mode");
    }

    if (params.ui_mcp_proxy) {
        ctx_http.get ("/cors-proxy",      ex_wrapper(proxy_handler_get));
        ctx_http.post("/cors-proxy",      ex_wrapper(proxy_handler_post));
        warn_names.push_back("MCP proxy (experimental)");
    } else {
        ctx_http.get ("/cors-proxy",      ex_wrapper(res_403));
        ctx_http.post("/cors-proxy",      ex_wrapper(res_403));
    }

    try {
        mcp_mgr.start(params);
    } catch (const std::exception & e) {
        SRV_ERR("MCP starting failed: %s\n", e.what());
        return 1;
    }

    if (!params.server_tools.empty() || !mcp_mgr.empty()) {
        try {
            tools.setup(params.server_tools, mcp_mgr, params.server_tools_runtime);
        } catch (const std::exception & e) {
            SRV_ERR("tools setup failed: %s\n", e.what());
            return 1;
        }
        ctx_http.get ("/tools",           ex_wrapper(tools.handle_get));
        ctx_http.post("/tools",           ex_wrapper(tools.handle_post));
        if (!params.server_tools.empty()) {
            warn_names.push_back("server tools (experimental)");
        }
        if (!params.server_tools_runtime.empty()) {
            warn_names.push_back("tools runtime (experimental)");
        }
        if (!mcp_mgr.empty()) {
            warn_names.push_back("MCP servers (experimental)");
        }
    } else {
        ctx_http.get ("/tools",           ex_wrapper(res_403));
        ctx_http.post("/tools",           ex_wrapper(res_403));
    }

    if (warn_names.size() > 0) {
        SRV_WRN("%s", "-----------------\n");
        SRV_WRN("%s", "the following feature(s) are enabled:\n");
        for (const auto & name : warn_names) {
            SRV_WRN("    %s\n", name.c_str());
        }
        SRV_WRN("%s", "do not expose the server to untrusted environments\n");
        SRV_WRN("%s", "-----------------\n");
    }

    //
    // Handle downloading model
    //

    if (child.is_child() && child.get_mode() == SERVER_CHILD_MODE_DOWNLOAD) {
        return child.run_download(params);
    } else if (!is_router_server && !is_run_by_cli) {
        // single-model mode (NOT spawned by router)
        // if this is invoked by CLI, model downloading should be already handled
        try {
            common_models_handler_apply(models_handler, params);
        } catch (const std::exception & e) {
            SRV_ERR("failed to download model: %s\n", e.what());
            return 1;
        }
    }

    //
    // Start the server
    //

    std::function<void()> clean_up;

    if (is_router_server) {
        SRV_INF("%s", "starting server in router mode. models will be automatically loaded on-demand\n");

        clean_up = [&models_routes, &mcp_mgr]() {
            SRV_INF("%s: cleaning up before exit...\n", __func__);
            // stop the session GC first, it finalizes live sessions and wakes pending readers
            server_stream_session_manager_stop();
            if (models_routes.has_value()) {
                models_routes->stopping.store(true); // maybe redundant, but just to be safe
                models_routes->models.unload_all();
            }
            mcp_mgr.shutdown();
            llama_backend_free();
        };

        if (!ctx_http.start()) {
            clean_up();
            SRV_ERR("%s", "exiting due to HTTP server error\n");
            return 1;
        }
        ctx_http.is_ready.store(true);

        shutdown_handler = [&](int) {
            if (models_routes.has_value()) {
                // important to disconnect any SSE clients
                models_routes->stopping.store(true);
            }
            mcp_mgr.shutdown();
            ctx_http.stop();
        };

        try {
            models_routes->models.load_startup_models();
        } catch (const std::exception & e) {
            SRV_ERR("failed to load models on startup: %s\n", e.what());
            ctx_http.stop();
            if (ctx_http.thread.joinable()) {
                ctx_http.thread.join();
            }
            clean_up();
            return 1;
        }

    } else {
        // setup clean up function, to be called before exit
        clean_up = [&ctx_http, &ctx_server, &mcp_mgr]() {
            SRV_INF("%s: cleaning up before exit...\n", __func__);
            // stop the session GC first, it finalizes live sessions and wakes pending readers
            server_stream_session_manager_stop();
            ctx_http.stop();
            ctx_server.terminate();
            mcp_mgr.shutdown();
            llama_backend_free();
        // detson inform python that we are quitting
        shm->buffers[0][0] = 424242.0f;
        sem_post(sem_c2p);

        };

        // start the HTTP server before loading the model to be able to serve /health requests
        if (!ctx_http.start()) {
            clean_up();
            SRV_ERR("%s", "exiting due to HTTP server error\n");
            return 1;
        }

        // setup communication child --> router if necessary
        if (child.is_child()) {
            ctx_server.set_state_callback([&](server_state state, json payload) {
                child.notify_to_router(server_state_to_str(state), payload);
            });
        }

        if (!ctx_server.load_model(params)) {
            clean_up();
            if (ctx_http.thread.joinable()) {
                ctx_http.thread.join();
            }
            SRV_ERR("%s", "exiting due to model loading error\n");
            return 1;
        }

        routes.update_meta(ctx_server);
        ctx_http.is_ready.store(true);

        SRV_INF("%s", "model loaded\n");

        shutdown_handler = [&](int) {
            mcp_mgr.shutdown();
            // this will unblock start_loop()
            ctx_server.terminate();
        };
    }

    // register signal handler if not running by CLI
    if (!is_run_by_cli) {
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
        struct sigaction sigint_action;
        sigint_action.sa_handler = signal_handler;
        sigemptyset (&sigint_action.sa_mask);
        sigint_action.sa_flags = 0;
        sigaction(SIGINT, &sigint_action, NULL);
        sigaction(SIGTERM, &sigint_action, NULL);
#elif defined (_WIN32)
        auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
            return (ctrl_type == CTRL_C_EVENT) ? (signal_handler(SIGINT), true) : false;
        };
        SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif
    }

    SRV_INF("listening on %s\n", ctx_http.listening_address.c_str());

    // TODO: remove this in the future
    // check the string to also handle the .sock case
    if (string_ends_with(ctx_http.listening_address, ":8080")) {
        SRV_WRN("%s", "NOTICE: server default port will be changed to :9931 in a future release\n");
        SRV_WRN("%s", "        ref: https://github.com/ggml-org/llama.cpp/pull/26508\n");
    }

    if (is_router_server) {
        if (!params.models_preset_hf.empty()) {
            SRV_WRN(      "NOTE: using preset.ini from HF repo '%s'\n", params.models_preset_hf.c_str());
            SRV_WRN("%s", "      please only use presets that you can trust! Unknown presets may be unsafe\n");
        }

        if (ctx_http.thread.joinable()) {
            ctx_http.thread.join(); // keep the main thread alive
        }

        // when the HTTP server stops, clean up and exit
        clean_up();
    } else {
        // optionally, notify router server that this instance is ready
        std::thread monitor_thread;
        if (child.is_child()) {
            monitor_thread = child.setup(shutdown_handler);
            child.notify_to_router(server_state_to_str(SERVER_STATE_READY), routes.get_model_info());
        }

        // this call blocks the main thread until queue_tasks.terminate() is called
        ctx_server.start_loop();

        clean_up();
        if (ctx_http.thread.joinable()) {
            ctx_http.thread.join();
        }
        if (monitor_thread.joinable()) {
            monitor_thread.join();
        }

        auto * ll_ctx = ctx_server.get_llama_context();
        if (ll_ctx != nullptr) {
            common_memory_breakdown_print(ll_ctx);
        }
    }

    return 0;
}
