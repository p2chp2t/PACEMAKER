# PACE: Towards Surfacing Hidden Conflicts in User Requests

**PaceMaker** is a conflict-aware retrieval and reasoning framework for personalized assistants.

PaceMaker retrieves relevant evidence from a user-specific knowledge base and reasons about whether a user request is feasible given the user's circumstances, commitments, and constraints.

## Quick Start

### 1. Set up the environment

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/p2chp2t/pacemaker
cd pacemaker

conda create -n pacemaker python
conda activate pacemaker
```

Install the required packages:

```bash
pip install -r requirements.txt
```

### 2. Build indices

```bash
python scripts/1_build_index.py \
    --data_dir <DATA_DIR> \
    --index_root <INDEX_DIR>
```

### 3. Run retrieval

```bash
python scripts/2_run_retrieval.py \
    --data_dir <DATA_DIR> \
    --index_root <INDEX_DIR> \
    --out_root <RETRIEVAL_OUTPUT_DIR> \
    --query_view_base_url <LLM_BASE_URL> \
    --filter_base_url <LLM_BASE_URL>
```

### 4. Generate answers

```bash
python scripts/3_run_answer.py \
    --data_dir <DATA_DIR> \
    --retrieval_root <RETRIEVAL_OUTPUT_DIR> \
    --profile_dir <PROFILE_DIR> \
    --out_root <ANSWER_OUTPUT_DIR> \
    --base_url <LLM_BASE_URL> \
    --api_key <API_KEY>
```

### 5. Run judge evaluation

```bash
python scripts/4_run_judge_eval.py \
    --data_dir <DATA_DIR> \
    --answer_root <ANSWER_OUTPUT_DIR> \
    --out_root <JUDGE_OUTPUT_DIR>
```

### 6. Evaluate and aggregate results

```bash
python scripts/5_evaluate.py \
    --data_dir <DATA_DIR> \
    --retrieval_root <RETRIEVAL_OUTPUT_DIR> \
    --judge_eval_root <JUDGE_OUTPUT_DIR> \
    --out_root <EVAL_OUTPUT_DIR>
```

Aggregate the results:

```bash
python scripts/6_aggregate_eval.py \
    --eval_root <EVAL_OUTPUT_DIR> \
    --out <AGGREGATE_OUTPUT_PATH>
```

## Dataset Schema

Each case is stored in a separate directory containing `corpus.json` and `queries.json`:

```text
<DATA_DIR>/
├── <case_id>/
│   ├── corpus.json
│   └── queries.json
└── ...
```

Persona profiles used for answer generation are stored separately:

```text
<PROFILE_DIR>/
├── <case_id>.json
└── ...
```

## Model and API Usage

The default configuration in the released code uses:

| Component         | Default Model                 |
| ----------------- | ----------------------------- |
| Embedding         | `Qwen/Qwen3-Embedding-8B`     |
| Retrieval agents  | `Qwen/Qwen3-4B-Instruct-2507` |
| Answer generation | `Qwen/Qwen3-4B-Instruct-2507` |
| Judge             | `gpt-5.4-mini`                |

When running the scripts:

* `<MODEL_NAME>` should be the model identifier exposed by your LLM server.
* `<LLM_BASE_URL>` should be the base URL of the API server hosting the LLM.
* `<API_KEY>` should be the authentication key expected by that server.
* `<YOUR_OPENAI_API_KEY>` is required when using OpenAI models directly.

The embedding module supports SentenceTransformers models, as well as the following OpenAI embedding models:

* `text-embedding-3-small`
* `text-embedding-3-large`
* `text-embedding-ada-002`

> **Note:** This repository contains the implementation used to run the PaceMaker pipeline. Some additional model configurations reported in the paper were evaluated using separate experimental code and are not included in the current release.

## Citation

If you find this work useful, please cite:

```bibtex
will be included soon
```
