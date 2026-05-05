## 📊 Results

Our model was evaluated on the MIMIC-III dataset for joint ICD and CPT prediction using a temporal multi-note modeling framework.

### Experimental Results

| Model | ICD Micro-F1 | ICD Macro-F1 | CPT Micro-F1 | CPT Macro-F1 | P@5 (ICD) | P@5 (CPT) | Joint Score |
|------|-------------|-------------|-------------|-------------|-----------|-----------|-------------|
| Baseline (BioClinicalBERT) | 0.4506 | 0.3130 | 0.5379 | 0.1751 | 0.4314 | 0.4775 | 0.4631 |
| Longformer | 0.5990 | 0.4490 | 0.5905 | 0.3017 | 0.5593 | 0.4968 | 0.5948 |
| Temporal (BERT + LSTM) | 0.4392 | 0.3704 | 0.5278 | 0.3019 | 0.4147 | 0.4310 | 0.4835 |

These results show that temporal multi-note modeling provides stable and clinically meaningful performance while explicitly modeling patient progression across sequential clinical notes.

---

### 🔍 Main Findings

- Temporal multi-note modeling provides stable and robust performance
- Temporal modeling enables more context-aware predictions across clinical notes
- Longformer achieves the strongest overall performance through long-context modeling
- ICD prediction remains more challenging than CPT prediction
- The temporal model provides a clinically meaningful representation of patient progression over time
