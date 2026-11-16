## Running retrieval experiments

Run all commands from the project root.

Before running the experiments, make sure that the FAISS dense indexes for KB1 and/or KB2 have already been created.

### Full retrieval evaluation

```bash
python -m src.experiments.retrieval_experiments --kb 1 --slice all
python -m src.experiments.retrieval_experiments --kb 2 --slice all
```

### Evaluation by query subset

Available slices:
- common, rare
- easy, medium, hard
- participant
- similarity
- narrative
- temporal
- spatial
- outcome
- weapon

Example
```bash
python -m src.experiments.retrieval_experiments --kb 2 --slice common
```
The script evaluates BM25, dense retrieval, and hybrid retrieval using Reciprocal Rank Fusion.

Metrics include:
- Recall@10
- Recall@20
- Recall@50
- MRR
- MAP@10
- nDCG@10

Results are saved to results/retrieval/

## Running oracle-generation experiments

Run all commands from the project root.

The experiment compares four configurations:

- Baseline LLM
- Raw-text-augmented LLM
- JSON KB1
- JSON KB2

The required inputs are:

- `data/queries/queries_all.json`
- `data/json_kb_v1/`
- `data/json_kb_v2/`
- `data/extracted_wikipedia_raw/`

An API key for the configured LLM must also be available through the project environment.

### Running Experiments

```bash
python -m src.experiments.oracle_generation_experiment --experiment overall
python -m src.experiments.oracle_generation_experiment --experiment rarity
python -m src.experiments.oracle_generation_experiment --experiment difficulty
python -m src.experiments.oracle_generation_experiment --experiment query-type
```

Select specific query-types
```bash
python -m src.experiments.oracle_generation_experiment \
  --experiment query-type \
  --types "Similarity" "Outcome-related"
  ```

Resuls are saved to results/oracle_generation/

## End-to-end RAG experiments

Run commands from the project root.

The experiment compares three generation settings:

- `baseline` — LLM without retrieved context
- `raw-text` — LLM with retrieved Wikipedia text
- `json` — LLM with retrieved JSON battle records

Hybrid retrieval results are used for the context-based settings.

### Examples

```bash
python -m src.experiments.e2e_rag_experiments --type baseline --slice all
python -m src.experiments.e2e_rag_experiments --type raw-text --top-k 20 --slice all
python -m src.experiments.e2e_rag_experiments --type json --top-k 20 --slice rare
python -m src.experiments.e2e_rag_experiments --type json --top-k 20 --slice narrative

Available slices:
all, common, rare, participant, similarity, narrative, temporal, spatial, outcome, weapon

Results are saved under results/e2e_rag/