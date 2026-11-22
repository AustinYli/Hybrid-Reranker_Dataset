# RAG Benchmark Harness

Repeatable retrieval and RAG evaluation. Track A benchmarks BM25, exact dense retrieval, hybrid RRF, and cross-encoder reranking on BEIR. Track B adds evidence retrieval, local answer generation, faithfulness, and abstention tests on MultiHop-RAG.

The fixed grid is in `results/track_a_runs.csv`. [RESULTS.md](RESULTS.md) has the findings, statistical caveats, and decision log.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Start with a small smoke test. Smoke runs write to `results/smoke_runs.csv`, not the main results table:

```bash
python -m rag_bench.run --config configs/scifact_bm25_smoke.json
# or:
scripts/run_smoke.sh
```

Then the Track A BM25 baselines:

```bash
python -m rag_bench.run --config configs/scifact_bm25.json
python -m rag_bench.run --config configs/nfcorpus_bm25.json
python -m rag_bench.run --config configs/fiqa_bm25.json
# or:
scripts/run_track_a_bm25.sh
```

SciFact BM25 analyzer and parameter check:

```bash
python scripts/run_bm25_gate.py
```

Dense and hybrid runs:

```bash
python -m rag_bench.run --config configs/scifact_dense_minilm.json
python -m rag_bench.run --config configs/scifact_hybrid_minilm.json
python -m rag_bench.run --config configs/scifact_rerank_minilm.json
python -m rag_bench.run --config configs/scifact_dense_minilm_rerank.json
python -m rag_bench.run --config configs/scifact_hybrid_minilm_rerank.json
```

Full grid plus the fusion sweeps:

```bash
python scripts/run_track_a_grid.py
python scripts/run_fusion_sweeps.py --force
python scripts/compare_track_a.py --samples 10000
python scripts/compare_rrf_k_sweep.py --samples 10000
python scripts/compare_fusion_winners.py --samples 10000
```

Reranker sweeps save the ranked lists and per-query metrics, record which Torch device was used, and skip runs that already finished. `rerank_latency_p50_ms` and `rerank_latency_p95_ms` are measured directly. The total query percentiles are labeled `component_percentile_sum` because they're the sum of the base and reranker percentiles:

```bash
python scripts/run_rerank_sweep.py \
  --reranker-model BAAI/bge-reranker-v2-m3 \
  --trust-remote-code \
  --datasets fiqa \
  --base-run-names fiqa_dense_bge_512 \
  --rerank-top-k-values 10,20,50,100

python scripts/compare_rerank_sweeps.py \
  --rerank results/rerank_bge_probe.csv \
  --output results/rerank_bge_probe_comparisons.csv \
  --holm-output results/rerank_bge_probe_comparisons_holm.csv \
  --include-adjacent-depths
```

## Notes

- BEIR documents are already passage-sized, so Track A doesn't test chunking.
- Embeddings and BM25 indexes are cached under `cache/`, so repeat runs are much faster.
- Dense retrieval uses exact NumPy inner product over L2-normalized embeddings by default. Set `retriever.index_backend` to `faiss` to use FAISS `IndexFlatIP`.
- E5 and BGE query/passage prefixes are added automatically based on the model name.
- Reranking is optional: it downloads a cross-encoder and is much slower than first-stage retrieval.
- Nothing is trained or fine-tuned here. The sweeps are tuned on BEIR test qrels, so the best settings they find are exploratory and should be checked on a separate split before real use.

## Useful Config Fields

- `dataset`: BEIR dataset name, for example `scifact`, `nfcorpus`, or `fiqa`
- `split`: usually `test`
- `max_corpus_docs` / `max_queries`: optional smoke-test limits
- `retriever.type`: `bm25`, `dense`, `hybrid`, or `rerank`
- `retriever.dense_model`: Sentence Transformers model name
- `retriever.index_backend`: `numpy` or `faiss`
- `retriever.analyzer`: `word-lower-v1`, `word-lower-stop-v1`, `word-lower-bigram-v1`, or `word-lower-stop-porter-v1`
- `retriever.k1` / `retriever.b` / `retriever.epsilon`: BM25 parameters
- `retriever.reranker_model`: CrossEncoder model name
- `retriever.max_seq_length`: truncation length for dense and reranker models
- `retriever.normalize_embeddings`: makes inner product the same as cosine similarity
- `metrics.cutoffs`: metric cutoffs, default `[1, 5, 10, 100]`

## Output

Each real run adds a row to `results/runs.csv` with the config, timings, artifact paths, and metrics such as:

- `ndcg@10`
- `recall@10`
- `mrr@10`
- `map@10`

`map@k` follows `pytrec_eval`'s `map_cut.k`: precision at each relevant rank up to `k`, divided by the total number of relevant documents for the query.

Timing fields:

- `load_seconds`: dataset loading, including any first-time download and unzip
- `index_seconds`: BM25 indexing, or embedding the corpus / loading it from cache
- `search_seconds`: query-time retrieval and reranking
- `throughput_ms_per_query`: `search_seconds / num_queries`
- `query_latency_p50_ms` and `query_latency_p95_ms`: per-query latency, when the retriever measures it

Ranked outputs are saved as JSONL under `results/artifacts/`, with per-query metric CSVs next to them.

If the metric code changes, recompute metrics from the saved rankings without rerunning retrieval:

```bash
python scripts/rescore_results.py results/track_a_runs.csv
```

## Comparing Runs

Paired bootstrap confidence intervals from the saved per-query metrics:

```bash
python -m rag_bench.compare \
  --baseline results/artifacts/<baseline>.per_query.csv \
  --candidate results/artifacts/<candidate>.per_query.csv \
  --metric ndcg@10 \
  --output results/comparisons.csv
```

The tests check nDCG, recall, and MAP against `pytrec_eval`, including a query with more relevant documents than the cutoff to cover the MAP denominator.

To run the same check over every saved Track A ranking:

```bash
python scripts/check_metrics_pytrec_eval.py
```

## Track B: MultiHop-RAG

Download and check the public files from the dataset's Hugging Face repo. The data goes in `data/` (git-ignored); the audit output is committed:

```bash
python scripts/inspect_multihop_rag.py
```

Run Stage 1 in this order. The reranker step checkpoints to `cache/` and picks up where it left off if interrupted:

```bash
python scripts/run_track_b_stage1.py --stage bm25
python scripts/run_track_b_stage1.py --stage dense
python scripts/run_track_b_stage1.py --stage hybrid
python scripts/run_track_b_stage1.py --stage rerank --rerank-batch-size 16
python scripts/compare_track_b_stage1.py --samples 10000
python scripts/check_track_b_evidence_metrics.py
python scripts/analyze_track_b_retrieval.py
```

The corpus is full news articles, so Track B splits them into 256-word chunks with 64-word overlap and indexes `title`, `source`, and `published_at` with every chunk. It reports both source-document recall and the stricter evidence-fact recall. `context_sufficiency@k` is 1 only when every annotated fact is in the selected chunks. That measures full coverage of the annotated evidence; it isn't a hard ceiling on accuracy, since a question might be answerable from evidence that wasn't annotated.

On MultiHop-RAG, sparse and dense swap places compared with FiQA: BGE-base beats BM25 by 0.1622 nDCG@10 on FiQA but trails it by 0.1645 evidence recall@5 here. Almost every answerable question names the publisher of its evidence, which gives BM25 exact keywords to match. The Track A gap law was tested here with settings fixed in advance, and it held: equal-weight `rrf_k=60` fusion raises dense evidence recall@5 by 0.0998 `[0.0900, 0.1098]`, more than any of the nine earlier fusion gains, but still ends up below BM25.

Install the local generator and run the oracle-context check before any generation with retrieved context:

```bash
ollama pull qwen2.5:7b-instruct-q4_K_M
python scripts/run_track_b_generation.py --mode oracle --max-queries 20
```

Once the full oracle check passes, run three repeats on the same fixed 200-query held-out sample used for retrieved-context evaluation, changing only the retrieval run:

```bash
python scripts/run_track_b_generation.py --mode retrieved \
  --retriever-run-name multihop_bm25_stem_w256_o64 \
  --evaluation-sample-size 200 --repeat 1
python scripts/run_track_b_generation.py --mode retrieved \
  --retriever-run-name multihop_bm25_stem_w256_o64 \
  --evaluation-sample-size 200 --repeat 2
python scripts/run_track_b_generation.py --mode retrieved \
  --retriever-run-name multihop_bm25_stem_w256_o64 \
  --evaluation-sample-size 200 --repeat 3

python scripts/analyze_track_b_determinism.py \
  results/track_b_generation_artifacts/<repeat-1>.jsonl \
  results/track_b_generation_artifacts/<repeat-2>.jsonl \
  results/track_b_generation_artifacts/<repeat-3>.jsonl \
  --output results/track_b_determinism.json

python scripts/run_track_b_generation.py \
  --mode retrieved \
  --retriever-run-name multihop_bm25_stem_w256_o64
```

The first 20 answerable questions were used to develop the prompt (their outputs were read while fixing the answer format), so full generation runs leave them out and record the ID-set hash and evaluation-partition version. Headline generation metrics use the remaining 2,235 answerable questions.

The full held-out oracle run scores 0.7955 EM and 0.8034 token F1. Three temperature-0 repeats on the fixed 200-query BM25 sample gave identical outputs (0.000 EM/F1 range), so runs don't need averaging. Oracle p50/p95 latency is 1.812/2.606 seconds per local request, run one at a time. The summary flags and ignores the per-component timings of 45 Ollama responses where they were impossible, but keeps every valid total request time.

Re-check format failures and error causes from the saved 200-query BM25 run without generating again:

```bash
python scripts/analyze_track_b_format.py \
  oracle=results/track_b_generation_artifacts/<oracle>.jsonl \
  bm25_sample200=results/track_b_generation_artifacts/<bm25-repeat>.jsonl
python scripts/analyze_track_b_generation.py \
  --generation-artifact results/track_b_generation_artifacts/<bm25-repeat>.jsonl \
  --allow-partial \
  --output results/track_b_failure_attribution_bm25_sample200.json
```

Badly formatted answers aren't what separates oracle from BM25: malformed BM25 answers on answerable questions still score 0.5714 EM and account for only 6 of 59 errors. Of the 59 errors, 50 are missing some annotated evidence in the prompt and 9 are wrong even with all of it. With no context at all, the model reaches only 0.2069 EM/F1 on the same sample, against 0.6609/0.6648 for BM25, so memory alone doesn't explain the retrieved-context results:

```bash
python scripts/run_track_b_generation.py \
  --mode closed_book --evaluation-sample-size 200 --repeat 1
```

On the matched 200-query pilot, EM is 0.6609 for BM25, 0.6322 for BGE dense, and 0.6897 for BGE dense plus the BGE reranker. Every aggregate interval crosses zero and none of the 24 aggregate/type EM/F1 comparisons survives Holm correction, so the pilot doesn't pick a winner. The 19–22% of questions where systems disagree does justify the full run: detecting the dense-to-reranker difference with 80% power at 0.05/24 needs about 960 answerable questions, and 2,235 are available.

After generating with BM25, dense, and dense plus reranker, compare the runs by name. The comparison covers aggregate, comparison, inference, and temporal results as one 24-test Holm family:

```bash
python scripts/compare_track_b_generation.py \
  --ordered-run-names <bm25-generation-run>,<dense-generation-run>,<reranker-generation-run>

python scripts/analyze_track_b_generation.py \
  --generation-artifact results/track_b_generation_artifacts/<run>.jsonl \
  --output results/track_b_failure_attribution.json
```

The position test uses three runs with the same retriever and chunk set, then one six-test Holm family:

```bash
python scripts/run_track_b_generation.py --mode retrieved \
  --retriever-run-name <retriever> --evidence-position first \
  --evaluation-sample-size 300
python scripts/run_track_b_generation.py --mode retrieved \
  --retriever-run-name <retriever> --evidence-position middle \
  --evaluation-sample-size 300
python scripts/run_track_b_generation.py --mode retrieved \
  --retriever-run-name <retriever> --evidence-position last \
  --evaluation-sample-size 300
python scripts/compare_track_b_positions.py \
  --ordered-run-names <first-run>,<middle-run>,<last-run>
```

The abstention test needs two retrieved runs, one with the abstention instruction off and one with it on. The script checks that the saved prompts differ only by that instruction, then corrects the null-question abstention test and the answerable false-abstention test together. Use the same 300-query sample for both runs:

```bash
python scripts/compare_track_b_abstention.py \
  --off-run-name <instruction-off-run> --on-run-name <instruction-on-run>
```

`--evaluation-sample-size` is only for the secondary experiments. It uses a versioned SHA-256 sampler and records the seed and the hash of the selected IDs. The full E1 retriever comparison doesn't use it.

The G1 context-depth sweep runs on one paired 300-query dev sample. Depth and the total context budget go up together:

```bash
python scripts/run_track_b_generation.py --mode retrieved \
  --retriever-run-name multihop_bm25_stem_w256_o64 \
  --retrieved-top-k 3 --max-context-words 768 --context-window 16384 \
  --evaluation-sample-size 300
python scripts/run_track_b_generation.py --mode retrieved \
  --retriever-run-name multihop_bm25_stem_w256_o64 \
  --retrieved-top-k 5 --max-context-words 1280 --context-window 16384 \
  --evaluation-sample-size 300
python scripts/run_track_b_generation.py --mode retrieved \
  --retriever-run-name multihop_bm25_stem_w256_o64 \
  --retrieved-top-k 10 --max-context-words 2560 --context-window 16384 \
  --evaluation-sample-size 300
python scripts/run_track_b_generation.py --mode retrieved \
  --retriever-run-name multihop_bm25_stem_w256_o64 \
  --retrieved-top-k 20 --max-context-words 5120 --context-window 16384 \
  --evaluation-sample-size 300

python scripts/compare_track_b_depths.py \
  --ordered-run-names <k3-run>,<k5-run>,<k10-run>,<k20-run>
```

The comparison reports how much evidence actually made it into each saved prompt alongside EM and F1, with separate 24-test Holm families for answer quality and evidence coverage. Full E1 runs must leave out the depth-selection sample with `--exclude-evaluation-sample-size 300 --exclude-evaluation-sample-seed 13`. The 16,384-token window was raised after a test at 8,192 tokens showed the k=20 prompts were being silently truncated; all four runs use the larger window.

G1 picks k=10. On 264 answerable questions, EM goes from 0.6742 at k=5 to 0.7500 at k=10, a paired +0.0758 `[0.0341, 0.1212]` that survives the 24-test Holm correction. k=20 raises context sufficiency from 0.2992 to 0.4545 but doesn't beat k=10 on EM (-0.0076 `[-0.0530, 0.0379]`) and doubles p50 latency from 14.36 to 29.23 seconds. This was chosen on the dev sample and isn't a final E1 number.

Each generation artifact records the exact context and its hash, the answer, the prompt hash, the context-builder and answer-parser versions, the hash of the retrieval run it used, the model tag and digest, quantization, runtime, seed, temperature, context window, the 96-token output limit, EM/F1, and whether it abstained. The NLI scorer outputs results across a range of thresholds and accepts a hand-labeled calibration CSV:

```bash
python scripts/score_track_b_faithfulness.py \
  --generation-artifact results/track_b_generation_artifacts/<run>.jsonl \
  --summary-output results/track_b_faithfulness_summary.json

python scripts/sample_faithfulness_calibration.py \
  --claim-artifact results/track_b_generation_artifacts/<run>.claims.jsonl
```

The calibration CSV leaves out the NLI score and includes the question, claim, and full context needed to fill in each `human_supported` label. Don't report a faithfulness rate until those 50 labels and Cohen's kappa are in. The first uncalibrated pass marks 0.6865 of oracle claims and 0.8734 of BM25 claims as unsupported at threshold 0.5. Treat these as diagnostics only: the oracle number is clearly too high, and the current setup checks each claim against one block at a time, so claims that combine facts from several blocks can't be supported. The saved threshold curves show this until calibration is done.

Retrieved generation defaults to five 256-word chunks and a hard 1,280-word context budget, counting block labels and retrieval metadata. Body words are split across the chunk set the same way regardless of display order. For chunking experiments, raise `--retrieved-top-k` and keep `--max-context-words 1280`. For the position test, repeat the same run with `--evidence-position first`, `middle`, and `last`.

E5 supports fixed, paragraph-based, and BGE-based semantic chunking. The non-fixed strategies need zero overlap. Semantic boundaries are saved and checked by hash. Run each retrieval config first, then use greedy packing in rank order so all four variants use the same total context budget:

```bash
python scripts/run_track_b_stage1.py --stage bm25 --overlap-words 0
python scripts/run_track_b_stage1.py --stage bm25 \
  --chunking-strategy structural --overlap-words 0
python scripts/run_track_b_stage1.py --stage bm25 \
  --chunking-strategy semantic --overlap-words 0

python scripts/run_track_b_generation.py --mode retrieved \
  --retriever-run-name <chunking-run> --retrieved-top-k 100 \
  --context-packing greedy --max-context-words 1280 \
  --evaluation-sample-size 300

python scripts/compare_track_b_chunking.py \
  --baseline-run-name <fixed-overlap-generation-run> \
  --candidate-run-names <no-overlap-run>,<structural-run>,<semantic-run>
```

[TRACK_B.md](TRACK_B.md) has the metric definitions, checks, and current results.
