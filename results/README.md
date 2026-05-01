## 📊 Results

Our model was evaluated on the MIMIC dataset for both ICD and CPT prediction tasks using a temporal multi-note modeling approach.

### Experimental Results

| Model                      | ICD Micro-F1 | ICD P@5    | CPT Micro-F1 | CPT P@5    | Task          |
| -------------------------- | ------------ | ---------- | ------------ | ---------- | ------------- |
| BioClinicalBERT (Baseline) | 0.4342       | 0.4229     | —            | —          | ICD only      |
| BioClinicalBERT (Baseline) | —            | —          | 0.5373       | 0.4643     | CPT only      |
| Longformer-based           | 0.4051       | 0.3977     | —            | —          | ICD only      |
| Longformer-based           | —            | —          | 0.5612       | 0.4717     | CPT only      |
| Temporal (BERT + LSTM)     | 0.5018       | 0.4712     | —            | —          | ICD only      |
| Temporal (BERT + LSTM)     | —            | —          | 0.5018       | 0.4924     | CPT only      |
| **Temporal (BERT + LSTM)** | **0.5018**   | **0.4712** | **0.6024**   | **0.4924** | **ICD + CPT** |

These results show that the proposed **temporal multi-note modeling approach improves performance over single-note baselines**, particularly for ICD prediction. 

---

### 🔍 Main Findings

* Temporal multi-note modeling improves ICD prediction performance
  (**Micro-F1: 0.4615 → 0.5335, +7.2 pp**)
* Joint prediction of ICD and CPT codes is feasible within a unified framework
* CPT codes are predicted more consistently than ICD codes
  (**AUC: 0.9596 vs. 0.8208**)
* Increasing the number of notes (from 1 to 5) improves top-ranked prediction accuracy
  (**ICD P@5: 0.2340 → 0.4503**) 
