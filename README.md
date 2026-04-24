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

This project develops a **temporal clinical reasoning framework** for automated ICD and CPT medical coding.

Unlike traditional NLP-based coding systems that treat each clinical note independently, our approach models **diagnostic changes over time**, integrates clinical notes across multiple time points, and aims to provide more consistent and explainable predictions.

Each clinical note is encoded using **BioClinicalBERT**, and an **LSTM** is used to capture temporal dependencies across notes. The model is trained in a multi-label setting to jointly predict diagnosis codes (**ICD**) and procedure codes (**CPT**).

---

## Task Definition

- **Input:** Sequential clinical notes + structured clinical data  
- **Output:** ICD-10 and CPT codes  
- **Task Type:** Multi-label classification  
- **Setting:** Admission-level prediction with temporal modeling  

---

## Data Access

Access to **MIMIC-IV** requires completion of the official data use training and credentialing process.

For this project, dataset access was obtained through **Steven Hu**, who completed the required training and certification to access MIMIC-IV.

The dataset includes:

- Clinical notes  
- ICD diagnosis codes  
- CPT procedure codes  
- Laboratory results  
- Medication records  

---

## Motivation

Automated medical coding is essential in healthcare because it supports billing, clinical decision-making, and large-scale medical research.

However, many existing NLP-based coding systems process clinical notes independently. This ignores the temporal evolution of a patient’s condition during admission, which can lead to incomplete understanding and inconsistent predictions.

Our project addresses this limitation by modeling multiple clinical notes over time.

---

## Objective

To build a **time-aware and context-aware auto-coding system** that:

- Tracks diagnostic evolution across admissions  
- Captures temporal relationships between clinical notes  
- Reduces false positives from outdated diagnoses  
- Supports joint ICD and CPT prediction  
- Produces more interpretable, evidence-based outputs  

---

## Proposed Approach

Our framework uses a temporal multi-note modeling pipeline:

1. Collect multiple clinical notes for each admission  
2. Sort notes chronologically based on chart time  
3. Encode each note using **BioClinicalBERT**  
4. Apply an **LSTM** to model temporal dependencies  
5. Build a shared admission-level representation  
6. Predict ICD and CPT codes using separate prediction heads  

---

## Preprocessing

We built a preprocessing pipeline that converts MIMIC clinical data into chronologically ordered admission-level sequences.

The preprocessing steps include:

- Selecting adult admissions with valid clinical notes and billing codes  
- Merging multiple note types such as discharge, radiology, nursing, physician, ECG, pharmacy, and respiratory notes  
- Sorting notes by chart time  
- Mapping ICD-9 codes to ICD-10 codes  
- Linking diagnosis and procedure codes to admissions  
- Splitting the dataset into train, validation, and test sets  

---

## Experimental Results

| Model | ICD Micro F1 | ICD P@5 | CPT Micro F1 | CPT P@5 | Task |
|------|-------------|--------|-------------|--------|------|
| Baseline 1 (BioClinicalBERT) | 0.4342 | 0.4229 | - | - | ICD only |
| Baseline 1 (BioClinicalBERT) | - | - | 0.5373 | 0.4643 | CPT only |
| Baseline 2 (Longformer-based) | 0.4051 | 0.3977 | - | - | ICD only |
| Baseline 2 (Longformer-based) | - | - | 0.5612 | 0.4717 | CPT only |
| Temporal Model (BERT + LSTM) | 0.5018 | 0.4712 | - | - | ICD only |
| Temporal Model (BERT + LSTM) | - | - | 0.5018 | 0.4924 | CPT only |
| Temporal Model (BERT + LSTM) | 0.5018 | 0.4712 | 0.6024 | 0.4924 | ICD + CPT |

---

## Main Findings

- Temporal multi-note modeling outperforms single-note baseline models.
- Joint ICD and CPT prediction is feasible within a single temporal framework.
- CPT codes are predicted more consistently than ICD codes.
- Increasing note coverage improves top-ranked prediction accuracy.
- Temporal modeling helps capture diagnostic progression across an admission.

---

## Comparison to Baseline

The baseline models process clinical notes independently or concatenate long text without explicitly modeling temporal progression.

In contrast, our approach models the sequence of clinical notes over time.

This allows the model to:

- Capture changes in patient condition  
- Improve consistency across notes  
- Reduce false positives from outdated diagnoses  
- Better represent admission-level clinical context  

---

## Limitations

- The model currently focuses on the most frequent ICD and CPT codes, which limits rare-code performance.
- Each admission is represented by a limited number of notes.
- The LSTM model may not fully capture long-range dependencies across long admissions.
- Structured clinical data such as labs and medications can be further integrated for stronger validation.

---

## Future Work

Future work will focus on:

- Integrating structured clinical data such as lab results and medications  
- Exploring Transformer-based temporal models  
- Improving rare-code prediction using class-weighted loss or focal loss  
- Conducting deeper error analysis  
- Improving interpretability of predicted ICD and CPT codes  

---

## Contributions

- **Woochang Shin**
  - Baseline Model 2: Longformer-based model
  - Temporal Model for joint ICD and CPT prediction
  - Project website development and maintenance
  - Project Poster and report writing support
    
  - **Jisun Kim**
  - Baseline Model 1: BioClinicalBERT note-level model
  - Temporal Model for ICD prediction
  - Project website maintenance and updates
  - Project documentation and report writing support  

- **Steven Hu**
  - MIMIC data access
  - Data preprocessing
  - Model training and execution (ICD-only, CPT-only, and joint ICD+CPT models)
  - Result extraction and evaluation
  - Project documentation and report writing support
    
- **Samarth Kumar Samal**
  - Model 1 (BioClinicalBERT) support  
  - Model 2 (Longformer) support  
  - Model 3 (Temporal / Joint Model) support  

---

## Summary

This project demonstrates that temporal multi-note modeling can improve automated medical coding by capturing clinical progression over time.

By moving beyond static note-level classification, our framework provides a more realistic and clinically meaningful approach for ICD and CPT code prediction.
