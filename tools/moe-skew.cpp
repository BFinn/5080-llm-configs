// Histogram MoE expert selection per layer, to measure routing skew.
// Captures the "ffn_moe_topk" tensor (selected expert ids) via the eval callback.
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <map>
#include <algorithm>

struct Acc {
    std::map<int, std::vector<long>> hist;   // layer -> per-expert counts
    long tokens = 0;
    int n_expert = 0;
    std::map<int,int> dumped;
};
static Acc A;

static bool cb_eval(struct ggml_tensor * t, bool ask, void * ud) {
    (void) ud;
    if (ask) return strncmp(t->name, "ffn_moe_argsort", 15) == 0;
    if (strncmp(t->name, "ffn_moe_argsort", 15) != 0) return true;
    // name is like "ffn_moe_topk-<il>"
    int il = -1; const char * d = strrchr(t->name, '-');
    if (d) il = atoi(d + 1);
    const int64_t ne0 = t->ne[0], ne1 = t->ne[1];   // [n_expert_used, n_tokens]
    static bool once = true;
    if (once) { once = false;
        fprintf(stderr, "[diag] name=%s type=%s ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu] view_src=%s cont=%d\n",
            t->name, ggml_type_name(t->type), (long long)t->ne[0],(long long)t->ne[1],(long long)t->ne[2],(long long)t->ne[3],
            (size_t)t->nb[0],(size_t)t->nb[1], t->view_src? "YES":"no", ggml_is_contiguous(t)); }
    // ffn_moe_argsort is the CONTIGUOUS full ranking [n_expert, n_tokens], best first.
    // The topk tensor is only a view of its first n_expert_used columns, and reading that
    // view yields valid data for the first rows only, so read the parent and slice here.
    if (!ggml_is_contiguous(t)) return true;
    const int TOPK = 10;
    std::vector<int32_t> buf((size_t) ne0 * ne1);
    ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * sizeof(int32_t));
    auto & h = A.hist[il];
    if (h.empty()) h.assign(A.n_expert, 0);
    if ((il == 0 || il == 24) && !A.dumped[il]) {
        A.dumped[il] = 1;
        const int64_t probe[] = {0,1,2,3,4,5,10,50,100,200,300,400,500,511};
        for (int64_t pi = 0; pi < (int64_t)(sizeof(probe)/sizeof(probe[0])); ++pi) { int64_t r = probe[pi]; if (r >= ne1) continue;
            fprintf(stderr, "[ids] layer %2d token %lld:", il, (long long) r);
            for (int c = 0; c < TOPK; ++c) fprintf(stderr, " %d", buf[(size_t) r * ne0 + c]);
            fprintf(stderr, "\n");
        }
    }
    for (int64_t r = 0; r < ne1; ++r)
        for (int c = 0; c < TOPK && c < ne0; ++c) {
            int32_t e = buf[(size_t) r * ne0 + c];
            if (e >= 0 && e < (int32_t) h.size()) h[e]++;
        }
    if (il == 0) A.tokens += ne1;
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <model.gguf> <n_expert> <text file> [n_tokens]\n", argv[0]); return 1; }
    const char * mpath = argv[1];
    A.n_expert = atoi(argv[2]);
    const char * tpath = argv[3];
    int want = argc > 4 ? atoi(argv[4]) : 2048;

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = getenv("SKEW_NGL") ? atoi(getenv("SKEW_NGL")) : 0;
    // mirror the serving config: experts on CPU, everything else on GPU
    static llama_model_tensor_buft_override ovr[2];
    if (getenv("SKEW_OT")) {
        ovr[0].pattern = "ffn_.*_exps";
        ovr[0].buft    = ggml_backend_cpu_buffer_type();
        ovr[1].pattern = nullptr; ovr[1].buft = nullptr;
        mp.tensor_buft_overrides = ovr;
    }
    fprintf(stderr, "[cfg] n_gpu_layers=%d expert_override=%s\n", mp.n_gpu_layers, getenv("SKEW_OT") ? "CPU" : "none");
    llama_model * model = llama_model_load_from_file(mpath, mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = want + 64; cp.n_batch = 512; cp.n_ubatch = 512;
    cp.cb_eval = cb_eval; cp.cb_eval_user_data = nullptr;
    cp.n_threads = 12; cp.n_threads_batch = 12;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return 1; }

    // tokenize REAL text: routing on synthetic ids would not reflect natural text
    std::string text;
    { FILE * f = fopen(tpath, "rb"); if (!f) { fprintf(stderr, "cannot open %s\n", tpath); return 1; }
      char b[65536]; size_t n;
      while ((n = fread(b, 1, sizeof b, f)) > 0 && text.size() < (size_t) want * 8) text.append(b, n);
      fclose(f); }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    // BUG FIXED: a buffer of want+64 overflowed for any real text; llama_tokenize then returns
    // a NEGATIVE required size and leaves the buffer untouched, so every run fed 512 zero tokens.
    std::vector<llama_token> toks(text.size() + 16);
    int nt = llama_tokenize(vocab, text.c_str(), (int) text.size(), toks.data(), (int) toks.size(), true, false);
    if (nt < 0) { fprintf(stderr, "tokenize overflow (%d)\n", nt); return 1; }
    toks.resize(std::min(nt, want));
    fprintf(stderr, "[tok] first ids:"); for (int i = 0; i < 8 && i < (int) toks.size(); ++i) fprintf(stderr, " %d", toks[i]); fprintf(stderr, "\n");
    fprintf(stderr, "tokenized %zu real tokens from %s\n", toks.size(), tpath);

    for (size_t off = 0; off < toks.size(); off += 512) {
        int n = (int) std::min((size_t) 512, toks.size() - off);
        if (llama_decode(ctx, llama_batch_get_one(toks.data() + off, n))) { fprintf(stderr, "decode failed\n"); break; }
    }

    printf("tokens=%ld  layers=%zu  n_expert=%d\n", A.tokens, A.hist.size(), A.n_expert);
    printf("layer  top1%%  top5%%  top10%%  top20%%  top50%%  gini  used/%d\n", A.n_expert);
    std::vector<double> agg_top10, agg_top20, agg_gini;
    for (auto & kv : A.hist) {
        std::vector<long> h = kv.second;
        long tot = 0; for (long v : h) tot += v;
        if (!tot) continue;
        int used = 0; for (long v : h) if (v) used++;
        std::sort(h.begin(), h.end(), std::greater<long>());
        auto frac = [&](double p){ size_t k = (size_t)(A.n_expert * p); long s = 0; for (size_t i = 0; i < k && i < h.size(); ++i) s += h[i]; return 100.0 * s / tot; };
        // gini
        std::vector<long> asc(h.rbegin(), h.rend());
        double cum = 0, gsum = 0; long n = asc.size();
        for (long i = 0; i < n; ++i) { cum += asc[i]; gsum += cum; }
        double gini = (n + 1 - 2.0 * gsum / tot) / n;
        printf("%5d  %5.1f  %5.1f  %6.1f  %6.1f  %6.1f  %.3f  %d\n",
               kv.first, frac(0.01), frac(0.05), frac(0.10), frac(0.20), frac(0.50), gini, used);
        agg_top10.push_back(frac(0.10)); agg_top20.push_back(frac(0.20)); agg_gini.push_back(gini);
    }
    { auto & h0 = A.hist[0];
      std::vector<std::pair<long,int>> v;
      for (size_t i = 0; i < h0.size(); ++i) if (h0[i]) v.push_back({h0[i], (int) i});
      std::sort(v.rbegin(), v.rend());
      long tot=0; for (auto&p:v) tot+=p.first;
      printf("\nlayer 0: %zu distinct experts used, total selections %ld\n", v.size(), tot);
      printf("  top 12 (id:count): "); for (size_t i=0;i<12&&i<v.size();++i) printf("%d:%ld ", v[i].second, v[i].first);
      printf("\n  rarest 6       : "); for (size_t i=(v.size()>6?v.size()-6:0);i<v.size();++i) printf("%d:%ld ", v[i].second, v[i].first);
      printf("\n"); }
    auto mean = [](std::vector<double>& v){ double s=0; for(double x:v) s+=x; return v.empty()?0:s/v.size(); };
    printf("\nMEAN across layers: top10%%=%.1f  top20%%=%.1f  gini=%.3f\n", mean(agg_top10), mean(agg_top20), mean(agg_gini));
    printf("uniform baseline : top10%%=10.0  top20%%=20.0  gini=0.000\n");
    llama_free(ctx); llama_model_free(model); llama_backend_free();
    return 0;
}
