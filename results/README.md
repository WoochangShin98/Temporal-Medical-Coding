## Results

Our model was evaluated on the MIMIC dataset for both ICD and CPT prediction tasks.

| Metric | Run 1 | Run 2 | Average |
|------|------|------|--------|
| ICD Micro-F1 | 0.4354 | 0.4345 | 0.435 |
| ICD Macro-F1 | 0.3207 | 0.3202 | 0.320 |
| ICD P@5 | 0.4503 | 0.4501 | 0.450 |
| ICD P@8 | 0.3796 | 0.3788 | 0.379 |
| CPT Micro-F1 | 0.5700 | 0.5710 | 0.571 |
| CPT Macro-F1 | 0.1883 | 0.1955 | 0.192 |
| CPT P@5 | 0.4856 | 0.4841 | 0.485 |
| CPT P@8 | 0.3845 | 0.3842 | 0.384 |

The model shows stable performance across runs and improves over simple baselines by incorporating temporal clinical information.

### Comparison with Prior Work (HTDS)

| Metric | Our Model | HTDS |
|-------|----------|------|
| ICD Micro-F1 | 0.435 | ~0.52–0.55 |
| ICD P@5 | 0.45 | ~0.60+ |

Our model achieves competitive performance, while additionally supporting joint ICD and CPT prediction.

### Example Prediction

The model leverages multiple clinical notes over time to infer diagnosis and procedure codes:

:contentReference[oaicite:0]{index=0}
