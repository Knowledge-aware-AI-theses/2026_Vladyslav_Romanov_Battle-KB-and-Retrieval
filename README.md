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