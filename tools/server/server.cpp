#include "server-context.h"
#include "server-http.h"

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "log.h"

#include <atomic>
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

static const char* SHM_NAME = "/ring_buffer_demo";
static const char* SEM_C2P = "/c2py_sem";
static const char* SEM_P2C = "/py2c_sem";
struct SharedMemory {
    float buffers[1][1000000];
};
SharedMemory *shm;
sem_t* sem_c2p;
sem_t* sem_py2c;

char **detsavelayer = (char **)malloc(sizeof(char *)*1000);
struct callback_data {
    std::vector<uint8_t> data;
};

static void detson_send_tensor(uint8_t * data, ggml_type type, const int64_t * ne, const size_t * nb) {
    float sum = 0;

    // Fill buffer
    int bufidx = 0;
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
                        GGML_ABORT("fatal error");
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
static bool detsoncb_save_embeds(struct ggml_tensor * t, bool ask, void * user_data) {
    if (ask) return true; // Always retrieve data
    const struct ggml_tensor * src0 = t->src[0];

    if (src0!=NULL && !strncmp(src0->name,"output.weight",13)) {
        auto * cb_data = (callback_data *) user_data;
        uint8_t * data = (uint8_t *) src0->data;
    
        // copy the data from the GPU memory if needed
        const bool is_host = ggml_backend_buffer_is_host(src0->buffer);
        printf("saving embeddings %d %d\n", is_host, ggml_is_quantized(src0->type));
        if (!is_host) {
            auto n_bytes = ggml_nbytes(src0);
            cb_data->data.resize(n_bytes);
            ggml_backend_tensor_get(src0, cb_data->data.data(), 0, n_bytes);
            printf("ERROR GPU not implemented yet");
        }       

        // save unembedding matrix
        if (src0->type != GGML_TYPE_F32) {
            auto nels = ggml_nelements(src0);
            printf("dequantizing... %d %d %d %d %d\n",nels,
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
            exit(1);
        }
    }
    return true;
}
static bool detsoncb_share_activs(struct ggml_tensor * t, bool ask, void * user_data) {
    // printf("detnode %s\n",t->name);
    if (ask) return true; // Always retrieve data
    auto * cb_data = (callback_data *) user_data;
    const struct ggml_tensor * src0 = t->src[0];
    const struct ggml_tensor * src1 = t->src[1];

    /*
    printf("detsonlayer %s %s %d %d %d %d\n",t->name, ggml_op_desc(t), t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
    if (src0!=NULL)
        printf("detsonSRC0 %s %s %d %d %d %d\n",src0->name, ggml_op_desc(src0), src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3]);
    if (src1!=NULL)
        printf("detsonSRC1 %s %s %d %d %d %d\n",src1->name, ggml_op_desc(src1), src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3]); 
    */

    // copy the data from the GPU memory if needed
    const bool is_host = ggml_backend_buffer_is_host(t->buffer);
    if (!is_host) {
        printf("detwarn cannot modify activations on GPU!\n");
        auto n_bytes = ggml_nbytes(t);
        cb_data->data.resize(n_bytes);
        ggml_backend_tensor_get(t, cb_data->data.data(), 0, n_bytes);
    }

    for (int i=0;i<1000;i++) {
        if (detsavelayer[i]==NULL) break;

        if (strlen(detsavelayer[i])==strlen(t->name)) {
if (!strncmp(t->name,detsavelayer[i],strlen(detsavelayer[i]))) {
                if (!ggml_is_quantized(t->type)) {
                    // printf("detson send %s %d %d %d\n",t->name,t->ne[0],t->ne[1],t->ne[2]);
                    uint8_t * data = is_host ? (uint8_t *) t->data : cb_data->data.data();
                    detson_send_tensor(data, t->type, t->ne, t->nb);
                }

                if (!strncmp(t->name,"result_norm",strlen("result_norm"))) {
                    // der layer, on modifie le token output selon la ladder
                    // la ladder a du deja modifier la shared RAM avec les nouvelles activations
                    if (shm->buffers[0][0]==42) {
                        // do not modify output
                    } else {
                        // recopie la shared RAM dans le computation graph de llamacpp
                        int bufidx = 2; // skip the 2 first ints = dims
                        uint8_t * data = (uint8_t *) t->data;
                        for (int64_t i3 = 0; i3 < t->ne[3]; i3++) {
                            for (int64_t i2 = 0; i2 < t->ne[2]; i2++) {
                                for (int64_t i1 = 0; i1 < t->ne[1]; i1++) {
                                    for (int64_t i0 = 0; i0 < t->ne[0]; i0++) {
                                        size_t i = i3 * t->nb[3] + i2 * t->nb[2] + i1 * t->nb[1] + i0 * t->nb[0];
                                        float *v = (float *) &data[i];
                                        v[0] = shm->buffers[0][bufidx++];
                                    }}}}
                    }
                }
            }
        }
    }
    return true;
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

// wrapper function that handles exceptions and logs errors
// this is to make sure handler_t never throws exceptions; instead, it returns an error response
static server_http_context::handler_t ex_wrapper(server_http_context::handler_t func) {
    return [func = std::move(func)](const server_http_req & req) -> server_http_res_ptr {
        std::string message;
        try {
            return func(req);
        } catch (const std::exception & e) {
            message = e.what();
        } catch (...) {
            message = "unknown error";
        }

        auto res = std::make_unique<server_http_res>();
        res->status = 500;
        try {
            json error_data = format_error_response(message, ERROR_TYPE_SERVER);
            res->status = json_value(error_data, "code", 500);
            res->data = safe_json_to_str({{ "error", error_data }});
            LOG_WRN("got exception: %s\n", res->data.c_str());
        } catch (const std::exception & e) {
            LOG_ERR("got another exception: %s | while hanlding exception: %s\n", e.what(), message.c_str());
            res->data = "Internal Server Error";
        }
        return res;
    };
}

int main(int argc, char ** argv) {
	// Create shared memory
    int fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
    ftruncate(fd, sizeof(SharedMemory));
    void* addr = mmap(nullptr, sizeof(SharedMemory), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    shm = reinterpret_cast<SharedMemory*>(addr);
    // Create semaphores
    sem_c2p = sem_open(SEM_C2P, O_CREAT, 0666, 0);
    sem_py2c = sem_open(SEM_P2C, O_CREAT, 0666, 0);

    // own arguments required by this example
    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER)) {
        return 1;
    }

    // TODO: should we have a separate n_parallel parameter for the server?
    //       https://github.com/ggml-org/llama.cpp/pull/16736#discussion_r2483763177
    // TODO: this is a common configuration that is suitable for most local use cases
    //       however, overriding the parameters is a bit confusing - figure out something more intuitive
    if (params.n_parallel == 1 && params.kv_unified == false && !params.has_speculative()) {
        LOG_WRN("%s: setting n_parallel = 4 and kv_unified = true (add -kvu to disable this)\n", __func__);

        params.n_parallel = 4;
        params.kv_unified = true;
    }

    common_init();

    // struct that contains llama context and inference
    server_context ctx_server;

    llama_backend_init();
    llama_numa_init(params.numa);

    // detson debug
    for (int i=0;i<1000;i++) detsavelayer[i]=NULL;
    callback_data cb_data;
    FILE *f = fopen("detembeds.bin","rb");
    if (f==NULL) {
        // detson save embeddings then exit
        params.cb_eval = detsoncb_save_embeds;
    } else {
        // detson pass the activations to python
        params.cb_eval = detsoncb_share_activs;
        fclose(f);
    }
    params.cb_eval_user_data = &cb_data;
    params.warmup = false;
    {
        int j=0;
        char line[10000];
        FILE *f = fopen("layers2save","r");
        if (f!=NULL) {
            while (fgets(line, sizeof(line), f) != NULL) {
                line[strlen(line)-1]=0; // -1 because we remove \n
                if (strlen(line)==0) break;
                detsavelayer[j]= (char *)malloc(sizeof(char)*strlen(line));
                strcpy(detsavelayer[j++],line);
            }
            fclose(f);
        }
    }


    LOG_INF("system info: n_threads = %d, n_threads_batch = %d, total_threads = %d\n", params.cpuparams.n_threads, params.cpuparams_batch.n_threads, std::thread::hardware_concurrency());
    LOG_INF("\n");
    LOG_INF("%s\n", common_params_get_system_info(params).c_str());
    LOG_INF("\n");

    server_http_context ctx_http;
    if (!ctx_http.init(params)) {
        LOG_ERR("%s: failed to initialize HTTP server\n", __func__);
        return 1;
    }

    //
    // Router
    //

    // register API routes
    server_routes routes(params, ctx_server, [&ctx_http]() { return ctx_http.is_ready.load(); });

    ctx_http.get ("/health",              ex_wrapper(routes.get_health)); // public endpoint (no API key check)
    ctx_http.get ("/v1/health",           ex_wrapper(routes.get_health)); // public endpoint (no API key check)
    ctx_http.get ("/metrics",             ex_wrapper(routes.get_metrics));
    ctx_http.get ("/props",               ex_wrapper(routes.get_props));
    ctx_http.post("/props",               ex_wrapper(routes.post_props));
    ctx_http.post("/api/show",            ex_wrapper(routes.get_api_show));
    ctx_http.get ("/models",              ex_wrapper(routes.get_models)); // public endpoint (no API key check)
    ctx_http.get ("/v1/models",           ex_wrapper(routes.get_models)); // public endpoint (no API key check)
    ctx_http.get ("/api/tags",            ex_wrapper(routes.get_models)); // ollama specific endpoint. public endpoint (no API key check)
    ctx_http.post("/completion",          ex_wrapper(routes.post_completions)); // legacy
    ctx_http.post("/completions",         ex_wrapper(routes.post_completions));
    ctx_http.post("/v1/completions",      ex_wrapper(routes.post_completions_oai));
    ctx_http.post("/chat/completions",    ex_wrapper(routes.post_chat_completions));
    ctx_http.post("/v1/chat/completions", ex_wrapper(routes.post_chat_completions));
    ctx_http.post("/api/chat",            ex_wrapper(routes.post_chat_completions)); // ollama specific endpoint
    ctx_http.post("/v1/messages",         ex_wrapper(routes.post_anthropic_messages)); // anthropic messages API
    ctx_http.post("/v1/messages/count_tokens", ex_wrapper(routes.post_anthropic_count_tokens)); // anthropic token counting
    ctx_http.post("/infill",              ex_wrapper(routes.post_infill));
    ctx_http.post("/embedding",           ex_wrapper(routes.post_embeddings)); // legacy
    ctx_http.post("/embeddings",          ex_wrapper(routes.post_embeddings));
    ctx_http.post("/v1/embeddings",       ex_wrapper(routes.post_embeddings_oai));
    ctx_http.post("/rerank",              ex_wrapper(routes.post_rerank));
    ctx_http.post("/reranking",           ex_wrapper(routes.post_rerank));
    ctx_http.post("/v1/rerank",           ex_wrapper(routes.post_rerank));
    ctx_http.post("/v1/reranking",        ex_wrapper(routes.post_rerank));
    ctx_http.post("/tokenize",            ex_wrapper(routes.post_tokenize));
    ctx_http.post("/detokenize",          ex_wrapper(routes.post_detokenize));
    ctx_http.post("/apply-template",      ex_wrapper(routes.post_apply_template));
    // LoRA adapters hotswap
    ctx_http.get ("/lora-adapters",       ex_wrapper(routes.get_lora_adapters));
    ctx_http.post("/lora-adapters",       ex_wrapper(routes.post_lora_adapters));
    // Save & load slots
    ctx_http.get ("/slots",               ex_wrapper(routes.get_slots));
    ctx_http.post("/slots/:id_slot",      ex_wrapper(routes.post_slots));

    //
    // Start the server
    //

    // setup clean up function, to be called before exit
    auto clean_up = [&ctx_http, &ctx_server]() {
        SRV_INF("%s: cleaning up before exit...\n", __func__);
        ctx_http.stop();
        ctx_server.terminate();
        llama_backend_free();
    };

    // start the HTTP server before loading the model to be able to serve /health requests
    if (!ctx_http.start()) {
        clean_up();
        LOG_ERR("%s: exiting due to HTTP server error\n", __func__);
        return 1;
    }

    // load the model
    LOG_INF("%s: loading model\n", __func__);

    if (!ctx_server.load_model(params)) {
        clean_up();
        if (ctx_http.thread.joinable()) {
            ctx_http.thread.join();
        }
        LOG_ERR("%s: exiting due to model loading error\n", __func__);
        return 1;
    }

    ctx_server.init();
    ctx_http.is_ready.store(true);

    LOG_INF("%s: model loaded\n", __func__);

    shutdown_handler = [&](int) {
        // this will unblock start_loop()
        ctx_server.terminate();
    };

    // TODO: refactor in common/console
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

    LOG_INF("%s: server is listening on %s\n", __func__, ctx_http.listening_address.c_str());
    LOG_INF("%s: starting the main loop...\n", __func__);
    // this call blocks the main thread until ctx_server.terminate() is called
    ctx_server.start_loop();

    clean_up();
    if (ctx_http.thread.joinable()) {
        ctx_http.thread.join();
    }
    llama_memory_breakdown_print(ctx_server.get_llama_context());

    return 0;
}
