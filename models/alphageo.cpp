struct Config : public BaseConfig
{
    int window_len;
    int max_distance;
    int num_buckets;
};

class Tokenizer : public BaseTokenizer
{
public:
    Tokenizer(const Config &config)
        : Tokenizer(config, nullptr)
    {}

    Tokenizer(const Config &config, BaseHistoryEncoder *encoder)
        : BaseTokenizer::BaseTokenizer(config, encoder)
    {
        sys_prompt = "";
        auto_add_bos = false;
    }

    size_t load(tokenizer::DataReader *buffer, int n_vocab) override
    {
        tp = new tokenizer::UnigramProcessor(3);
        size_t size = tp->Load(buffer, n_vocab);
        comma_tok_id = tp->PieceToId(" ;");

        terminate_ids.insert(comma_tok_id);

        tp->RegisterPreprocessor(new tokenizer::TextPrepAddLeadingSpace());
        return size;
    }

public:
    int comma_tok_id;
};

static void ggml_compute_forward_meliad_kq_norm_f32(ggml::tensor * dst , const ggml::tensor * a, int ith, int nth, void *userdata)
{
    const struct ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const size_t nb01 = src0->nb[1];
    const size_t nb02 = src0->nb[2];
    const size_t nb03 = src0->nb[3];

    const size_t nb1 = dst->nb[1];
    const size_t nb2 = dst->nb[2];
    const size_t nb3 = dst->nb[3];

    const float eps = 1e-6f;

    // TODO: optimize
    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            for (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                const float * x = (float *) ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);

                double sum = 0.0f;
                for (int64_t i00 = 0; i00 < ne00; i00++) {
                    sum += (float)(x[i00] * x[i00]);
                }

                float *y = (float *)((char *)dst->data + i01*nb1 + i02*nb2 + i03*nb3);

                const double scale = 1.0f / sqrt(sum + eps);

                for (int64_t i00 = 0; i00 < ne00; i00++) {
                    y[i00] = (float)(x[i00] * scale);
                }
            }
        }
    }
}

static void ggml_compute_forward_meliad_kq_norm(ggml::tensor * dst , const ggml::tensor * a, int ith, int nth, void *userdata)
{
    const ggml::tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            ggml_compute_forward_meliad_kq_norm_f32(dst, a, ith, nth, userdata);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

struct rel_pos_diag_mask_params
{
    int window_len;
    int n_past;
    float value;
};

static void ggml_compute_forward_rel_pos_diag_mask_inf_f32(ggml::tensor * dst , const ggml::tensor * a, int ith, int nth, const rel_pos_diag_mask_params *param)
{
    const int n  = (int)ggml_nrows(a);
    const int nc = (int)a->ne[0];
    const int nr = (int)a->ne[1];
    const int nz = (int)n/nr;

    const int n_past = param->n_past;
    const float value = param->value;

    GGML_ASSERT(dst->nb[0] == sizeof(float));
    GGML_ASSERT(  a->nb[0] == sizeof(float));

    for (int k = 0; k < nz; k++) {
        for (int j = ith; j < nr; j += nth) {
            for (int i = 0; i < nc; i++) {
                if ((i >= n_past + j) || (i < n_past - param->window_len)) {
                    *(float *)((char *) dst->data + k*dst->nb[2] + j*dst->nb[1] + i*dst->nb[0]) = value;
                }
                else {
                    *(float *)((char *) dst->data + k*dst->nb[2] + j*dst->nb[1] + i*dst->nb[0]) = *(float *)((char *) a->data + k*a->nb[2] + j*a->nb[1] + i*a->nb[0]);
                }
            }
        }
    }
}

static void ggml_compute_forward_rel_pos_diag_mask_inf(ggml::tensor * dst , const ggml::tensor * a, int ith, int nth, void *userdata)
{
    const ggml::tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            ggml_compute_forward_rel_pos_diag_mask_inf_f32(dst, a, ith, nth, (const rel_pos_diag_mask_params *)userdata);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

struct safe_softmax_param
{
    float min_x;
};

static void ggml_compute_forward_safe_soft_max_f32(ggml::tensor * dst , const ggml::tensor * a, int ith, int nth, const safe_softmax_param *param)
{
    const struct ggml_tensor * src0 = a;

    const int nc = (int)a->ne[0];
    const int nr = (int)ggml::nrows(a);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr * ith;
    const int ir1 = MIN(ir0 + dr, nr);

    const float min_x = param->min_x;

    for (int i1 = ir0; i1 < ir1; i1++) {
        float * sp = (float *)((char *) src0->data + i1*src0->nb[1]);
        float * dp = (float *)((char *)  dst->data +  i1*dst->nb[1]);

        float x_max = min_x;
        for (int i0 = 0; i0 < nc; i0++) {
            if (sp[i0] > x_max)
                x_max = sp[i0];
        }

        float x_sum = 0.0f;
        for (int i0 = 0; i0 < nc; i0++) {
            dp[i0] = expf(sp[i0] - x_max);
            x_sum += dp[i0];
        }

        const float scale = 1.0f / (x_sum + expf(min_x - x_max));
        for (int i0 = 0; i0 < nc; i0++) {
            dp[i0] *= scale;
        }
    }
}

static void ggml_compute_forward_safe_soft_max(ggml::tensor * dst , const ggml::tensor * a, int ith, int nth, void *userdata)
{
    const ggml::tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            ggml_compute_forward_safe_soft_max_f32(dst, a, ith, nth, (const safe_softmax_param *)userdata);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

class AlphaGeoSelfAttention: public BaseAttention
{
public:
    AlphaGeoSelfAttention(InitContext *ctx, int hidden_size, int num_attention_heads, int max_length, int max_distance, int num_buckets)
        : BaseAttention(ctx, hidden_size, num_attention_heads, num_attention_heads, hidden_size / num_attention_heads, max_length, false, false,
                        max_length),
          rel_embedding(ggml::new_tensor_2d(ctx, ggml::type::GGML_TYPE_F16, num_attention_heads, num_buckets)),
          attention_scale(ggml::new_tensor_1d(ctx, ggml::type::GGML_TYPE_F32, num_attention_heads)),
          pos_bucket(nullptr)
    {
        diag_mask_params.value = -1000000.0f;
        softmax_params.min_x = -1000.0f;
        set_prec(ggml::prec::GGML_PREC_F32);
    }

    int64_t get_param_num(bool effective_only) const override
    {
        int64_t r = BaseAttention::get_param_num(effective_only);
        r += ggml::nelements(rel_embedding);
        r += ggml::nelements(attention_scale);
        return r;
    }

protected:
    float get_attn_scale(int index) const
    {
        const float *p = (const float *)attention_scale->data;
        return p[index];
    }

    // output: [heads, qlen, head_size]
    ggml::tensor *get_k_from_cache(ComputeContext *ctx, const int hidden_size, const int n_past, const int qlen) override
    //ggml::tensor *get_k_from_cachexxx(ComputeContext *ctx, const int hidden_size, const int n_past, const int qlen)
    {
        const int head_size = k_hidden_size / num_kv_heads;

        ggml::tensor *key_layer = nullptr;

        if (n_past > 0)
        {
            key_layer = ggml::view_1d(ctx, k_cache, n_past * k_hidden_size, 0);
            key_layer = ggml::reshape_3d(ctx, key_layer, head_size, num_kv_heads, n_past);  // [n_past, heads, head_size]
        }
        else
        {
            ggml::fill(k_cache, 0, 0, 2 * k_hidden_size * ggml::element_size(v_cache));

            // an empty tensor (all 0)
            key_layer = ggml::view_1d(ctx, k_cache, 1 * k_hidden_size, 1 * ggml::row_size(k_cache));
            key_layer = ggml::reshape_3d(ctx, key_layer, head_size, num_kv_heads, 1);       // [1, heads, head_size]
        }

        key_layer = ggml::permute(ctx, key_layer, 0, 2, 1, 3);                              // [heads, qlen, head_size]

        return key_layer;
    }

    // output: [heads, head_size, klen]
    ggml::tensor *get_v_from_cache(ComputeContext *ctx, const int hidden_size, const int n_past, const int qlen) override
    //ggml::tensor *get_v_from_cachexxx(ComputeContext *ctx, const int hidden_size, const int n_past, const int qlen)
    {
        const int head_size = v_hidden_size / num_kv_heads;

        ggml::tensor * value_layer = nullptr;

        if (n_past > 0)
        {
            value_layer = ggml::view_3d(ctx,
                        v_cache,
                        n_past, head_size, num_kv_heads,
                        cache_length * ggml::element_size(v_cache),
                        cache_length * ggml::element_size(v_cache) * head_size,
                        0); // [n_past, head_size, klen]
        }
        else
        {
            // an empty tensor (all 0)
            ggml::fill(v_cache, 0);
            value_layer = ggml::view_3d(ctx,
                        v_cache,
                        1, head_size, num_kv_heads,
                        cache_length * ggml::element_size(v_cache),
                        cache_length * ggml::element_size(v_cache) * head_size,
                        1 * ggml::element_size(v_cache)); // [1, head_size, klen]
        }

        return value_layer;
    }

    ggml::tensor *build_pos_bias(ComputeContext *ctx, ggml::tensor *pos_bucket, ggml::tensor *attn_rel_b) const
    {
        ggml::tensor *pos_bucket_1d = ggml::view_1d(ctx, pos_bucket, pos_bucket->ne[0] * pos_bucket->ne[1], 0);

        ggml::tensor *pos_bias = ggml::get_rows(ctx, attn_rel_b, pos_bucket_1d);

        pos_bias = ggml::view_3d(ctx, pos_bias,
                    pos_bias->ne[0], pos_bucket->ne[0], pos_bucket->ne[1],
                    ggml::element_size(pos_bias) * pos_bias->ne[0],
                    ggml::element_size(pos_bias) * pos_bias->ne[0] * pos_bucket->ne[0],  0);

        pos_bias = ggml::permute(ctx, pos_bias, 2, 0, 1, 3);

        pos_bias = ggml::cont(ctx, pos_bias);

        return pos_bias;
    }

    // k: [heads, qlen, head_size]
    // q: [heads, qlen, head_size]
    // v: [heads, head_size, klen]
    ggml::tensor *calc_attn_scores(ComputeContext *ctx, int hidden_size, const int n_past, const int qlen,
                                            ggml::tensor *key_layer, ggml::tensor *query_layer, ggml::tensor *value_layer) override
    {
        key_layer   = ggml::map_custom1(ctx, key_layer, ggml_compute_forward_meliad_kq_norm, GGML_N_TASKS_MAX, nullptr);
        query_layer = ggml::map_custom1(ctx, query_layer, ggml_compute_forward_meliad_kq_norm, GGML_N_TASKS_MAX, nullptr);

        // note auto-broadcasting in ggml_mul_mat for `repeat > 1`
        ggml::tensor *attn_scores = ggml::mul_mat(ctx, key_layer, query_layer); // [heads, qlen, klen]

        // default to F32 here
        ggml::mul_mat_set_prec(attn_scores, GGML_PREC_F32);

        attn_scores = apply_pos_embedding_kq(ctx, attn_scores, hidden_size, qlen, pos);

        // attn_masked = mask_past(attn_scores)
        diag_mask_params.n_past = n_past;
        ggml::tensor * attn_masked = ggml::map_custom1(ctx, attn_scores, ggml_compute_forward_rel_pos_diag_mask_inf, GGML_N_TASKS_MAX, &diag_mask_params);

        // attn_probs = soft_max(attn_masked)
        ggml::tensor * attn_probs = ggml::map_custom1(ctx, attn_masked, ggml_compute_forward_safe_soft_max, GGML_N_TASKS_MAX, &softmax_params);

        ggml::tensor *context_layer = ggml::mul_mat(ctx, value_layer, attn_probs); // [heads, qlen, head_size]

        ggml::mul_mat_set_prec(context_layer, GGML_PREC_F32);

        last_attn_scores = ggml::reshape_2d(ctx,
            ggml::cont(ctx, ggml::permute(ctx, context_layer, 0, 2, 1, 3)),
            hidden_size, qlen);

        return last_attn_scores;
    }

    ggml::tensor *apply_pos_embedding_kq(ComputeContext *ctx, ggml::tensor *kq, int hidden_size, int qlen, ggml::tensor *past) const override
    {
        ggml::tensor *bias = build_pos_bias(ctx, pos_bucket, rel_embedding);
        kq = ggml::add(ctx, kq, bias);

        ggml::tensor *att_scale_view = ggml::view_3d(ctx, attention_scale, 1, 1, attention_scale->ne[0],
            1 * ggml::element_size(attention_scale), 1 * 1 * ggml::element_size(attention_scale), 0);

        kq = ggml::mul(ctx, kq, att_scale_view);

        return kq;
    }

public:
    ggml::tensor *rel_embedding;
    ggml::tensor *attention_scale;
    ggml::tensor *pos_bucket;
    rel_pos_diag_mask_params diag_mask_params;
    safe_softmax_param softmax_params;

    //int n_past;
};

class AGMLP : public TheMLP
{
public:
    AGMLP() = default;
    AGMLP(InitContext *ctx, int hidden_size)
        : AGMLP(ctx, hidden_size, 4 * hidden_size) {}

    AGMLP(InitContext *ctx, int hidden_size, int intermediate_size)
        : TheMLP(ctx, hidden_size, intermediate_size, ActFunc::RELU, false) {}
};

class AlphaGeoLMBlock : public Block
{
public:
    AlphaGeoLMBlock() = default;

    AlphaGeoLMBlock(InitContext *ctx, int hidden_size, int num_attention_heads, int intermediate_size,
                int max_length, int max_distance, int num_buckets)
        : input_layernorm(ctx, hidden_size),
          attention(ctx, hidden_size, num_attention_heads, max_length, max_distance, num_buckets),
          post_attention_layernorm(ctx, hidden_size),
          mlp(ctx, hidden_size, intermediate_size)
    {
        mlp.set_prec(ggml::prec::GGML_PREC_F32);
    }

    using Block::forward;
    ggml::tensor *forward(ComputeContext *ctx, ggml::tensor *hidden_states, int n_past) override
    {
        ggml::tensor *input = hidden_states;

        hidden_states = input_layernorm.forward(ctx, hidden_states);

        ggml::tensor *attn_outputs = attention.forward(ctx, hidden_states, n_past);

        attn_outputs = ggml::add(ctx, attn_outputs, input);

        input = attn_outputs;

        hidden_states = post_attention_layernorm.forward(ctx, attn_outputs);

        hidden_states = mlp.forward(ctx, hidden_states);

        hidden_states = ggml::add(ctx, hidden_states, input);

        return hidden_states;
    }

    void shift_cache(int shift, int total) override
    {
        attention.shift_cache(shift, total);
    }

    void set_id(int id) override
    {
        Block::set_id(id);

        post_attention_layernorm.set_id(id);
        input_layernorm.set_id(id);
        attention.set_id(id);
        mlp.set_id(id);
    }

    int64_t get_param_num(bool effective_only) const override
    {
        int64_t r = 0;
        r += input_layernorm.get_param_num(effective_only);
        r += attention.get_param_num(effective_only);
        r += post_attention_layernorm.get_param_num(effective_only);
        r += mlp.get_param_num(effective_only);
        return r;
    }

    size_t get_cache_size(void) const override
    {
        return attention.get_cache_size();
    }

    void  set_cache_buffer(BackendBuffer *buffer) override
    {
        return attention.set_cache_buffer(buffer);
    }

    size_t read_cache_data(void *buffer, size_t buffer_size) const override
    {
        return attention.read_cache_data(buffer, buffer_size);
    }

    size_t write_cache_data(const void *buffer, size_t buffer_size) override
    {
        return attention.write_cache_data(buffer, buffer_size);
    }

public:
    RMSNorm input_layernorm;
    AlphaGeoSelfAttention attention;
    RMSNorm post_attention_layernorm;
    AGMLP mlp;
};

static int relative_position_bucket(int x, int y, const int max_distance, int n_buckets, bool bidirectional)
{
    if (bidirectional) {
        n_buckets >>= 1;
    }

    const int64_t max_exact = n_buckets >> 1;

    int32_t relative_position = x - y;
    int32_t relative_bucket = 0;
    if (bidirectional) {
        relative_bucket += (relative_position > 0) * n_buckets;
        relative_position = abs(relative_position);
    } else {
        relative_position = -std::min<int32_t>(relative_position, 0);
    }
    int32_t relative_position_if_large = (int32_t)floorf(max_exact + logf(1.0f * relative_position / max_exact) * (n_buckets - max_exact) / logf(1.0f * max_distance / max_exact));
    relative_position_if_large = std::min<int32_t>(relative_position_if_large, n_buckets - 1);
    relative_bucket += (relative_position < max_exact ? relative_position : relative_position_if_large);
    return relative_bucket;
}

class ModelClass : public Model<Config, Embedding, RMSNorm, AlphaGeoLMBlock, int, int, int, int, int, int>
{
public:
    ModelClass(InitContext *ctx, const Config &config, Block *lm_head, int hidden_size, int num_attention_heads,
               int intermediate_size, int max_length, int max_distance, int num_buckets)
        : Model(ctx, config, lm_head, hidden_size, num_attention_heads,
                intermediate_size, max_length, max_distance, num_buckets),
          config(config)
    {
        pos_bucket = ggml::new_tensor_2d(ctx, GGML_TYPE_I32, max_length, max_length);
        ctx->get_allocator()->alloc(pos_bucket);
    }

protected:
    void before_forward(ComputeContext *ctx, ggml::tensor *input_ids, int n_past) override
    {
        Model::before_forward(ctx, input_ids, n_past);

        const int n_tokens = (int)input_ids->ne[0];
        const int n_kv = n_past > 0 ? n_past : 1;

        ggml::tensor *view = ggml::view_2d(ctx, pos_bucket, n_kv, n_tokens,
            n_kv * ggml::element_size(pos_bucket), 0);

        int32_t       *data = (      int32_t *)pos_bucket->data;

        for (int j = 0; j < n_tokens; ++j)
        {
            for (int i = 0; i < n_kv; ++i)
            {
                data[j * n_kv + i] = relative_position_bucket(i, n_past + j, config.max_distance, config.num_buckets, false);
            }
        }

        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            auto &attention = layers[i].attention;
            attention.pos_bucket = view;
            attention.diag_mask_params.window_len = config.window_len;
        }
    }
protected:
    ggml::tensor *pos_bucket;
    const Config config;
};

class ConditionalGeneration : public BaseModelForConditionalGeneration
{
public:
    ConditionalGeneration() = default;
    ConditionalGeneration(const Config &config, const RuntimeConfig &runtime_config, ModelType type = ModelType::MODEL_TYPE_ALPHAGEO_LM)
        : BaseModelForConditionalGeneration(type, config, runtime_config), config(config)
    {
        const size_t tensor_ovhd = ggml_tensor_overhead();
        const size_t num_tensors = 3 + config.num_hidden_layers * 13;
        const size_t ctx_size = num_tensors * tensor_ovhd;
        w_ctx_.gctx = GGMLContext({.mem_size = ctx_size, .mem_buffer = nullptr, .no_alloc = true});
        w_ctx_.dtype = config.dtype;
        w_ctx_.cache_dtype = ggml::type::GGML_TYPE_F32;

        CHATLLM_CHECK(config.dtype == ggml::type::GGML_TYPE_F32) << "this model must be quantized to F32";

        transformer = new ModelClass(&w_ctx_, config, nullptr,
                                    config.hidden_size, config.num_attention_heads,
                                    config.intermediate_size, config.max_length, config.max_distance, config.num_buckets);
        batch_input = false;
    }

    void load(ModelLoader &loader) override
    {
        auto transformer = get_typed_transformer<ModelClass>();

        transformer->word_embeddings->load("model.embed_tokens.", &loader);

        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            std::string layer_prefix = "model.layers." + std::to_string(layer_ids[i]) + '.';

            loader.read_tensor(layer_prefix + "rel_embedding.weight",               transformer->layers[i].attention.rel_embedding);
            loader.read_tensor(layer_prefix + "self_attn.attention_scale.weight",   transformer->layers[i].attention.attention_scale);

            loader.read_tensor(layer_prefix + "input_layernorm.weight",             transformer->layers[i].input_layernorm.weight);
            loader.read_tensor(layer_prefix + "mlp.hidden0.weight",                 transformer->layers[i].mlp.fc0.weight);
            loader.read_tensor(layer_prefix + "mlp.output_layer.weight",            transformer->layers[i].mlp.fc1.weight);
            loader.read_tensor(layer_prefix + "post_attention_layernorm.weight",    transformer->layers[i].post_attention_layernorm.weight);

            loader.read_tensor(layer_prefix + "self_attn.k_proj.weight", transformer->layers[i].attention.k_proj.weight);
            loader.read_tensor(layer_prefix + "self_attn.o_proj.weight", transformer->layers[i].attention.o_proj.weight);
            loader.read_tensor(layer_prefix + "self_attn.q_proj.weight", transformer->layers[i].attention.q_proj.weight);
            loader.read_tensor(layer_prefix + "self_attn.v_proj.weight", transformer->layers[i].attention.v_proj.weight);
        }

        transformer->final_layernorm->load("model.norm.", &loader);

        CHATLLM_CHECK(w_ctx_.get_used_mem() == w_ctx_.get_mem_size())
            << "corrupted model weights";
    }

    bool is_output_terminated(const std::vector<int> &output_ids, int &keep_idx, int &pop_output) override
    {
        if (output_ids.size() < 1)
            return false;

        int last_tok_id = output_ids[output_ids.size() - 1];

        Tokenizer *tok = dynamic_cast<Tokenizer *>(tokenizer);

        if (tok->comma_tok_id == last_tok_id)
        {
            pop_output = 0;
            return true;
        }
        else
        {
            keep_idx = (int)output_ids.size();
            return false;
        }
    }

public:
    Config config;
};