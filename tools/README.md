# tools

## moe-skew.cpp — measure MoE expert-routing skew

Captures the router's `ffn_moe_argsort` tensor through llama.cpp's eval callback and
histograms which experts each token actually selects. Answers "is routing skewed enough
to exploit?" on *your* model and *your* text, which decides whether an expert cache is
worth building.

```bash
g++ -O2 -std=c++17 -I$L/include -I$L/ggml/include -I$L/common -o moe-skew moe-skew.cpp \
    -L$L/build/bin -lllama -lggml -lggml-base -Wl,-rpath,$L/build/bin
SKEW_NGL=99 SKEW_OT=1 ./moe-skew model.gguf 512 some_text.txt 512
#   args: <gguf> <n_expert> <text file> [n_tokens]; SKEW_OT=1 puts experts on CPU as in serving
```

Three bugs were found and fixed getting this to produce real numbers, each of which
produced a confident, wrong answer. They are worth knowing if you write your own:

1. `ffn_moe_topk` is a **non-contiguous view** whose row stride spans all experts. Read
   contiguously it returns full permutations and reports *perfectly uniform* routing
   (Gini exactly 0.000). Read the contiguous parent `ffn_moe_argsort` and slice the top-k.
2. That fix still only reads the first rows correctly if you assume the buffer is
   compact. Use the tensor's own `nb[1]` stride or read the parent.
3. `llama_tokenize` returns a **negative** required size on buffer overflow and leaves the
   buffer untouched. A 576-token buffer on real text silently fed 512 zero-tokens to the
   model and reported extreme, input-independent skew (Gini 0.96). Size the buffer to the
   text. The control that caught it: two different inputs must not route identically.

Always run a control with two different texts before believing a routing measurement.
