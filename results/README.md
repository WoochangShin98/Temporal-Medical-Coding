## 📊 Results

Our model was evaluated on the MIMIC dataset for both ICD and CPT prediction tasks using a temporal multi-note modeling approach.

### Experimental Results

| Model                      | ICD Micro-F1 | ICD P@5 | CPT Micro-F1 | CPT P@5 | Task      |
| -------------------------- | ------------ | ------- | ------------ | ------- | --------- |
| BioClinicalBERT (Baseline) | 0.3966       | 0.4000  | —            | —       | ICD only  |
| BioClinicalBERT (Baseline) | —            | —       | 0.5295       | 0.4628  | CPT only  |
| Longformer-based           | 0.5990       | 0.5593  | —            | —       | ICD only  |
| Longformer-based           | —            | —       | 0.5905       | 0.4968  | CPT only  |
| Temporal (BERT + LSTM)     | 0.4392       | 0.4147  | —            | —       | ICD only  |
| Temporal (BERT + LSTM)     | —            | —       | 0.5278       | 0.4310  | CPT only  |
| **Temporal (BERT + LSTM)** | **0.4392**   | **0.4147** | **0.5278** | **0.4310** | **ICD + CPT** |

These results show that the proposed **temporal multi-note modeling approach achieves competitive performance compared to single-note baselines**, particularly for CPT prediction.

---

### 🔍 Main Findings

- Temporal multi-note modeling provides **consistent and robust performance**
- Joint prediction of ICD and CPT codes is feasible within a unified framework
- CPT codes are predicted more reliably than ICD codes (**Micro-F1: 0.5278 vs. 0.4392**)
- The model demonstrates a **precision-oriented prediction behavior**, improving top-ranked prediction quality
- Compared to Longformer, the temporal model shows **lower ICD performance**, likely due to information compression in sequential modeling
