# PACE: Towards Surfacing Hidden Conflicts in User Requests

**PaceMaker** is a conflict-aware retrieval and reasoning framework for personalized assistants.

PaceMaker retrieves relevant evidence from a user-specific knowledge base and reasons about whether a user request is feasible given the user's circumstances, commitments, and constraints.

## Quick Start

### Set up the environment

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

### 0. Download the dataset

PACE is available on [Hugging Face](https://huggingface.co/datasets/p2chp2t/pace).

Download and prepare the dataset:

```bash
python scripts/0_load_data.py --output_dir data/pace
```

This creates the following directory structure:

```text
data/pace/
├── kb/
│   ├── <case_id>/
│   │   ├── corpus.json
│   │   └── queries.json
│   └── ...
└── profile/
    ├── <case_id>.json
    └── ...
```

### 1. Build indices

```bash
python scripts/1_build_index.py \
    --data_dir data/pace/kb \
    --index_root <INDEX_DIR>
```

### 2. Run retrieval

For local LLMs, start an API server (e.g., with vLLM) and provide its base URL through `--query_view_base_url` and `--filter_base_url`.

```bash
python scripts/2_run_retrieval.py \
    --data_dir data/pace/kb \
    --index_root <INDEX_DIR> \
    --out_root <RETRIEVAL_OUTPUT_DIR> \
    --query_view_base_url <LLM_BASE_URL> \
    --filter_base_url <LLM_BASE_URL>
```

To use OpenAI models directly:

```bash
export OPENAI_API_KEY=<OPENAI_API_KEY>

python scripts/2_run_retrieval.py \
    --data_dir data/pace/kb \
    --index_root <INDEX_DIR> \
    --out_root <RETRIEVAL_OUTPUT_DIR> \
    --query_view_model <OPENAI_MODEL> \
    --filter_model <OPENAI_MODEL>
```

### 3. Generate answers

For a local LLM:

```bash
python scripts/3_run_answer.py \
    --data_dir data/pace/kb \
    --retrieval_root <RETRIEVAL_OUTPUT_DIR> \
    --profile_dir data/pace/profile \
    --out_root <ANSWER_OUTPUT_DIR> \
    --base_url <LLM_BASE_URL> \
    --api_key <LLM_API_KEY>
```

To use an OpenAI model directly:

```bash
export OPENAI_API_KEY=<OPENAI_API_KEY>

python scripts/3_run_answer.py \
    --data_dir data/pace/kb \
    --retrieval_root <RETRIEVAL_OUTPUT_DIR> \
    --profile_dir data/pace/profile \
    --out_root <ANSWER_OUTPUT_DIR> \
    --model <OPENAI_MODEL>
```

### 4. Run judge evaluation

```bash
export OPENAI_API_KEY=<OPENAI_API_KEY>

python scripts/4_run_judge_eval.py \
    --data_dir data/pace/kb \
    --answer_root <ANSWER_OUTPUT_DIR> \
    --out_root <JUDGE_OUTPUT_DIR>
```

### 5. Evaluate the results

```bash
python scripts/5_evaluate.py \
    --data_dir data/pace/kb \
    --retrieval_root <RETRIEVAL_OUTPUT_DIR> \
    --judge_eval_root <JUDGE_OUTPUT_DIR> \
    --out_root <EVAL_OUTPUT_DIR>
```

### 6. Aggregate results

```bash
python scripts/6_aggregate_eval.py \
    --eval_root <EVAL_OUTPUT_DIR> \
    --out <AGGREGATE_OUTPUT_PATH>
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

* `<LLM_BASE_URL>` should be the base URL of the API server hosting the LLM.
* `<LLM_API_KEY>` should be the authentication key expected by that server.
* `<OPENAI_MODEL>` should be the OpenAI model used for retrieval or answer generation.
* `<OPENAI_API_KEY>` should be your OpenAI API key.

The embedding module supports SentenceTransformers models, as well as the following OpenAI embedding models:

* `text-embedding-3-small`
* `text-embedding-3-large`
* `text-embedding-ada-002`

> **Note:** This repository contains the implementation used to run the PaceMaker pipeline. Some additional model configurations reported in the paper were evaluated using separate experimental code and are not included in the current release.

## Citation

Citation information will be added soon.
