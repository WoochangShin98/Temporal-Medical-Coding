# Temporal Clinical Reasoning for Medical Coding  
CSCI 5541 – Natural Language Processing (Team Project)

---

## Team Members
- Woochang Shin  
- Jisun Kim  
- Steven Hu  
- Samarth Kumar Samal  

---

## Project Overview

This project proposes a **temporal clinical reasoning framework** for automated ICD and CPT code prediction using electronic health records (EHRs).

Unlike traditional approaches that process clinical notes independently, our method models **patient progression over time** by integrating multiple clinical notes across an admission.

Each clinical note is encoded using **BioClinicalBERT**, and a **bidirectional LSTM** is used to capture temporal dependencies across notes.  
The model is trained in a multi-label setting to jointly predict diagnosis (**ICD**) and procedure (**CPT**) codes.

---

## Task Definition

- **Input:** Chronologically ordered multi-note clinical events  
- **Output:** ICD-10 and CPT codes  
- **Task Type:** Multi-label classification  
- **Setting:** Admission-level prediction with temporal modeling  

---

## Data

We use the **MIMIC-III dataset**, a large-scale ICU dataset containing de-identified clinical records.

The dataset includes:

- Clinical notes  
- ICD diagnosis codes  
- Laboratory results  
- Medication records  

We selected MIMIC-III because it preserves diverse clinical note types (e.g., nursing, physician, radiology), which are essential for temporal modeling.

---

## Motivation

Automated medical coding is critical in healthcare, supporting billing, clinical decision-making, and large-scale research.

However, most existing systems process clinical notes independently, ignoring how a patient’s condition evolves over time.

This leads to incomplete understanding and inconsistent predictions.

Our work addresses this limitation by modeling **temporal relationships across multiple clinical notes**.

---

## Proposed Approach

Our framework consists of three main components:

### 1. Note Encoding
Each clinical note is encoded using **BioClinicalBERT**, producing contextual embeddings.

### 2. Temporal Modeling
Chronologically ordered embeddings are passed through a **bidirectional LSTM** to capture temporal dependencies and patient progression.

### 3. Joint Prediction
The shared temporal representation is used to predict **ICD and CPT codes simultaneously** using multi-label classification.

---

## Preprocessing

We construct admission-level sequences from MIMIC-III:

- Select adult patients (≥ 24 hours admission)
- Merge heterogeneous note types (nursing, radiology, physician, etc.)
- Sort notes by chart time
- Normalize text and remove PHI placeholders
- Map ICD-9 codes to ICD-10
- Extract CPT codes from CPTEVENTS
- Truncate to first 5 events for temporal modeling
- Split dataset into train/validation/test (80/10/10)

---

## Experimental Results

| Model | ICD Micro-F1 | ICD Macro-F1 | CPT Micro-F1 | CPT Macro-F1 | P@5 (ICD) | P@5 (CPT) | Joint |
|------|-------------|-------------|-------------|-------------|-----------|-----------|-------|
| Baseline | 0.3966 | 0.1947 | 0.5295 | 0.1758 | 0.4000 | 0.4628 | 0.4631 |
| Longformer | 0.5990 | 0.4490 | 0.5905 | 0.3017 | 0.5593 | 0.4968 | 0.5948 |
| Temporal (BERT + LSTM) | 0.4392 | 0.3704 | 0.5278 | 0.3019 | 0.4147 | 0.4310 | 0.4835 |

---

## Main Findings

- Temporal modeling improves consistency by capturing patient progression
- Multi-note modeling outperforms single-note baselines in interpretability
- Longformer achieves the strongest overall performance due to long-context modeling
- ICD prediction is more challenging than CPT prediction
- Temporal modeling provides a clinically meaningful representation of patient history

---

## Limitations

- Performance is biased toward frequent codes (rare-code problem)
- Limited note coverage (max 5 events)
- LSTM struggles with long-range dependencies
- Joint prediction introduces task trade-offs (ICD vs CPT)

---

## Future Work

- Explore Transformer-based temporal models
- Improve rare-code prediction (e.g., focal loss)
- Incorporate structured data (labs, medications)
- Perform deeper error analysis
- Improve interpretability of predictions

---

## Summary

This project demonstrates that **temporal multi-note modeling** improves automated medical coding by capturing clinical progression over time.

By moving beyond static note-level approaches, our framework provides a more realistic and clinically meaningful solution for ICD and CPT prediction.

---

## Contributions

- **Woochang Shin**
  - **Model 1**: Initial and final development of Baseline Model 2 (Longformer) for clinical code prediction
  - **Model 3**: Initial and final development of the temporal model for joint ICD and CPT prediction
  - Project website development and maintenance
  - Project poster, documentation

- **Jisun Kim**
  - **Model 2**: Initial and final development of Baseline Model 1 (BioClinicalBERT) note-level model
  - **Model 3**:Initial and final development of the temporal model for joint ICD and CPT prediction
  - Project website maintenance and updates
  - Project report, documentation

- **Steven Hu**
  - MIMIC data access and preprocessing
  - Model training and execution (ICD-only, CPT-only, and joint ICD+CPT)
  - Result extraction and evaluation
  - Project report, documentation
    
- **Samarth Kumar Samal**
  - Model 1 (BioClinicalBERT) for joint ICD and CPT Prediction
  - Model 2 (Longformer) for joint ICD and CPT Prediction
  - Model 3 (Temporal / Joint Model) for joint ICD and CPT Prediction Support
  - Project Support
