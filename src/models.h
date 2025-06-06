#pragma once

#include <ggml.h>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <stdint.h>

#include "tokenizer.h"
#include "chat.h"
#include "layers.h"

namespace chatllm
{
    std::string to_string(ModelPurpose purpose);
    std::string format_model_capabilities(uint32_t model_type);

    class ModelBlock: public Block
    {
    public:
        virtual int save_session(FILE *f) = 0;
        virtual int load_session(FILE *f) = 0;

        virtual int save_session(ModelSessionMemory &session) const = 0;
        virtual int load_session(ModelSessionMemory &session) = 0;

        virtual void load(const std::string &path, TensorLoader *loader, const std::vector<int> &layer_ids) = 0;
    };

    class HeterogeneousModel;

    class ModelFinalSteps
    {
    public:
        virtual ggml::tensor *forward(HeterogeneousModel *model, ComputeContext *ctx, ggml::tensor *input_ids, ggml::tensor *hidden_states)
        {
            return nullptr;
        }
    };

    class HeterogeneousModel : public ModelBlock
    {
    public:
        HeterogeneousModel() = default;

        HeterogeneousModel(InitContext *ctx, int num_hidden_layers, int hidden_size,
            Block *word_embeddings, Block *final_layernorm,
            Block *lm_head, std::function<Block *(InitContext *, int)> create_layer);

        ~HeterogeneousModel();

        ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *input_ids, int n_past) override;
        void set_ctx(int n_ctx) override;

        void shift_cache(int shift, int total) override;

        int64_t get_param_num(bool effective_only) const override;

        Block *get_layer(int index);

        void set_final_steps(std::unique_ptr<ModelFinalSteps> final_steps);
        ModelFinalSteps *get_final_steps();

        int save_session(FILE *f) override;
        int load_session(FILE *f) override;
        int save_session(ModelSessionMemory &session) const override;
        int load_session(ModelSessionMemory &session) override;
        void load(const std::string &path, TensorLoader *loader, const std::vector<int> &layer_ids) override;
    private:
        struct state
        {
            size_t cache_size;
        };
    protected:
        virtual void before_forward(ComputeContext *ctx, ggml::tensor *input_ids, int n_past) {}

        virtual int64_t get_param_num_of_layers(bool effective_only) const;

    public:
        const int num_hidden_layers;
        const int hidden_size;
        Block *word_embeddings;
        Block *final_layernorm;
        Block *lm_head;
        Block *logits_pp;
    protected:
        // std::vector<std::unique_ptr<Block>> layers;
        std::vector<Block *> layers;
        size_t cache_size;
        std::unique_ptr<ModelFinalSteps> final_steps;
    };

    class LMFinalSteps : public ModelFinalSteps
    {
    public:
        ggml::tensor *forward(HeterogeneousModel *model, ComputeContext *ctx, ggml::tensor *input_ids, ggml::tensor *hidden_states) override;
    };

    class EmbeddingPoolingFinalSteps : public ModelFinalSteps
    {
    public:
        ggml::tensor *forward(HeterogeneousModel *model, ComputeContext *ctx, ggml::tensor *input_ids, ggml::tensor *hidden_states) override;
    };

    class EmbeddingLastTokenFinalSteps : public ModelFinalSteps
    {
    public:
        ggml::tensor *forward(HeterogeneousModel *model, ComputeContext *ctx, ggml::tensor *input_ids, ggml::tensor *hidden_states) override;
    };

} // namespace chatllm
