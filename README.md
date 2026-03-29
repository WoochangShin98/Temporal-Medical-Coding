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

This project develops a temporal clinical reasoning framework for automated ICD and CPT medical coding. 

Unlike traditional NLP-based coding systems that treat each clinical note independently, our approach models diagnostic changes over time, integrates structured clinical data with free-text notes, and aims to provide more consistent and explainable predictions.

---
## Task Definition

- Input: Sequential clinical notes + structured data (labs, meds)
- Output: ICD-10 and CPT codes (multi-label classification)
- Setting: Admission-level prediction with temporal modeling

---

## Data Access

Access to **MIMIC-IV** requires completion of the official data use training and credentialing process.

For this project, dataset access was obtained through Steven Hu, who has completed the required training and certification to access MIMIC-IV.

- Clinical notes  
- ICD diagnosis codes  
- CPT procedure codes  
- Laboratory results  
- Medication records  

---

## Objective

To build a time-aware and context-aware auto-coding system that:

- Tracks diagnostic evolution across admissions  
- Detects documentation inconsistencies  
- Produces interpretable, evidence-based outputs  

---

## Preliminary Results

We implemented an initial baseline using ClinicalBERT on individual notes.

| Model | Micro F1 |
|------|--------|
| Baseline (ClinicalBERT, note-level) | TBD |
| Proposed (partial temporal model) | TBD |

Initial experiments are in progress. Early observations suggest:
- Note-level models struggle with longitudinal consistency
- Temporal aggregation may improve stability

---
## Comparison to Baseline

The baseline model processes each note independently, which ignores temporal relationships.

Our approach aims to:
- Improve consistency across notes
- Reduce false positives from outdated diagnoses

Preliminary experiments will evaluate whether temporal modeling improves F1-score and reduces inconsistency errors.

---
## Next Steps

- Implement temporal modeling across notes
- Integrate structured data (lab results, medications)
- Perform hyperparameter tuning
- Conduct ablation study (temporal vs structured vs full model)

---
## Plan and Feedback

Current progress aligns with the proposed direction.

Next milestone:
- Complete baseline implementation
- Validate temporal modeling improvements

We will refine the model based on experimental results and instructor feedback.
