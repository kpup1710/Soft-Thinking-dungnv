# Soft Thinking — dungnv Fork

This is a fork of the official [Soft Thinking](https://github.com/eric-ai-lab/Soft-Thinking) implementation, extended with additional experiments on **small models (Qwen3-1.7B)** and a **projection-based concept token** mechanism.

---

## What Is Soft Thinking?

**Soft Thinking** ([arXiv 2505.15778](https://arxiv.org/abs/2505.15778)) is a technique that improves LLM reasoning by replacing discrete token sampling during the "thinking" phase with **weighted sums over token embeddings**.

### Standard vs. Soft Thinking

| Phase | Standard Generation | Soft Thinking |
|---|---|---|
| Logits → probs | softmax | softmax |
| Token selection | argmax / sample one token | keep top-k tokens + their probs |
| Next input | `embedding[token]` | `sum(prob[i] * embedding[token[i]])` for i in top-k |
| After `</think>` | — | reverts to standard discrete sampling |

The model "thinks" in a continuous concept space rather than committing to a single discrete token at each step, which gives it more expressive power during multi-step reasoning.

### Key Parameters

| Parameter | Recommended | Description |
|---|---|---|
| `--max_topk` | 10 (range 5–20) | Number of top tokens to blend |
| `--min_p` | 0.001 (range 0.0–0.005) | Minimum probability threshold |
| `--early_stopping_entropy_threshold` | 0.01 (range 0.0–0.1) | Stop thinking early if entropy is below this |
| `--early_stopping_length_threshold` | 256 (range 256–1024) | Minimum thinking length before early stopping |

---

## What This Fork Adds

### 1. Small-Model Experiments (Qwen3-1.7B)

The original paper focuses on large models (QwQ-32B). This fork adds scripts to test Soft Thinking on **Qwen3-1.7B** on a single GPU.

**`scripts/st/qwen3_1.7b_st.sh`** — standard Soft Thinking on Qwen3-1.7B:

```bash
python run_sglang_softthinking.py \
    --dataset "aime2024" \
    --model_name "./models/Qwen/Qwen3-1.7B" \
    --max_topk 10 \
    --min_p 0.001 \
    --early_stopping_entropy_threshold 0.01 \
    --early_stopping_length_threshold 256 \
    --num_gpus 1 \
    --num_samples 1 \
    --enable_soft_thinking
```

> **Note:** Per the original paper's warning, Soft Thinking yields suboptimal results on models ≤7B (and sometimes ≤14B) because the limited hidden size places the last hidden state near unrelated embeddings, introducing noise during probability weighting. These small-model experiments are exploratory.

---

### 2. Projection-Based Concept Token (`--use_projection_concept_token`)

**`scripts/st/qwen3_1.7b_st_projection.sh`** adds a new flag that orthogonally projects each weighted concept embedding onto the **unembedding subspace** before feeding it into the next layer.

**Motivation:** In small models, the raw weighted embedding (sum of top-k token embeddings weighted by probability) may fall far from the model's natural input distribution. Projecting it onto the unembedding subspace keeps the concept token closer to a "plausible word vector", potentially reducing the noise problem on small models.

```bash
python run_sglang_softthinking.py \
    --dataset "aime2024" \
    --model_name "./models/Qwen/Qwen3-1.7B" \
    --max_topk 10 \
    --min_p 0.001 \
    --early_stopping_entropy_threshold 0.01 \
    --early_stopping_length_threshold 256 \
    --num_gpus 1 \
    --num_samples 1 \
    --enable_soft_thinking \
    --use_projection_concept_token
```

---

## Repository Layout

```
Soft-Thinking/
├── datasets/                  # Benchmark datasets (AIME, MATH500, GSM8k, GPQA, LiveCodeBench, ...)
├── models/
│   ├── download.py            # HuggingFace model downloader
│   └── Qwen/Qwen3-1.7B/      # Downloaded model weights (this fork)
├── scripts/
│   ├── baseline/
│   │   └── qwq32b.sh          # Baseline (no soft thinking) on QwQ-32B
│   └── st/
│       ├── qwq32b_st_math.sh          # Soft Thinking on QwQ-32B, math benchmarks
│       ├── qwq32b_gumble.sh           # Soft Thinking + Gumbel/Dirichlet noise
│       ├── qwen3_1.7b_st.sh           # [THIS FORK] Soft Thinking on Qwen3-1.7B
│       └── qwen3_1.7b_st_projection.sh # [THIS FORK] + projection concept token
├── sglang_soft_thinking_pkg/  # Modified SGLang v0.4.6.post1 (Apache 2.0)
│   └── python/sglang/srt/
│       ├── configs/model_config.py       # soft_thinking config params
│       ├── layers/sampler.py             # Core weighted sampling logic
│       ├── layers/vocab_parallel_embedding.py  # weighted_forward() method
│       └── models/deepseek_v2.py         # DeepSeek-V2 model adaptation
├── run_sglang_softthinking.py # Main inference entry point
├── run_sglang_nothinking.py   # Baseline inference (standard sampling)
├── matheval.py                # Math answer evaluation
├── codeeval.py                # Code execution evaluation
├── configure.sh               # Environment setup
└── readme.md                  # Original upstream README
```

---

## Environment Setup

```bash
conda create -n st python=3.11 -y && conda activate st
pip install --upgrade pip
pip install torch transformers accelerate jsonlines math_verify openai torch_memory_saver
pip install flash_attn --no-build-isolation  # may take ~20 min

# Install the customized SGLang
cd sglang_soft_thinking_pkg
pip install -e "python[all]"
cd ..
```

### Docker (Recommended for Reproducibility)

Results vary across devices due to floating-point precision differences. The original experiments used NVIDIA H100.

```bash
docker run -it --name h100_st --gpus all \
    --shm-size 32g --network host \
    -v /.cache:/root/.cache \
    -v <path_to_workspace>:/workspace \
    --env "HF_TOKEN=<your_hf_token>" \
    --ipc=host \
    lmsysorg/sglang:latest /bin/bash
```

---

## Quick Start

```bash
# Download model
python ./models/download.py --model_name "Qwen/Qwen3-1.7B"

# Run Soft Thinking (small model, single GPU)
bash scripts/st/qwen3_1.7b_st.sh

# Run with projection concept token
bash scripts/st/qwen3_1.7b_st_projection.sh
```

For the full QwQ-32B experiments (original paper), see `readme.md`.

---

## Supported Benchmarks

| Category | Datasets |
|---|---|
| Math reasoning | AIME2024, AIME2025, MATH500, GSM8k, GPQA Diamond, AMC23 |
| Code generation | HumanEval, MBPP, LiveCodeBench |

> For coding benchmarks, run inference **without** `--reeval` first, then run again **with** `--reeval` for evaluation (required due to a multiprocessing limitation).

---

## Random Perturbation Extensions

Noise-injection variants of Soft Thinking (from [arXiv 2508.03440](https://arxiv.org/abs/2508.03440)):

```bash
python run_sglang_softthinking.py \
    ... \
    --add_noise_gumbel_softmax \
    --gumbel_softmax_temperature 0.5 \
    --add_noise_dirichlet \
    --dirichlet_temperature 1.0
```

See `scripts/st/qwq32b_gumble.sh` for a full example.

---

## Licensing

- **This fork's additions** (files outside `sglang_soft_thinking_pkg/`): **MIT License**
- **Modified SGLang** (`sglang_soft_thinking_pkg/`): **Apache License 2.0** (derivative of SGLang v0.4.6.post1)

---

## Citation

```bibtex
@article{zhang2025soft,
  title={Soft Thinking: Unlocking the Reasoning Potential of LLMs in Continuous Concept Space},
  author={Zhang, Zhen and He, Xuehai and Yan, Weixiang and Shen, Ao and Zhao, Chenyang and Wang, Shuohang and Shen, Yelong and Wang, Xin Eric},
  journal={arXiv preprint arXiv:2505.15778},
  year={2025}
}
```
