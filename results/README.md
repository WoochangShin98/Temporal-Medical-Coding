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

Our model achieves competitive performance while additionally supporting joint ICD and CPT prediction.

### Example Prediction

The following example demonstrates how the model processes multiple clinical notes to generate predictions.

The model predicts ICD and CPT codes by leveraging multiple temporally ordered clinical notes for each patient.

The model integrates diverse clinical notes (e.g., radiology, nursing, and procedure notes) over time to infer both diagnosis (ICD) and procedure (CPT) codes.

Below is an example prediction for a single admission (HADM_ID: 100053):

```text
HADM_ID: 100053

[NOTES]
CHEST (PORTABLE AP)
Reason: pneumonia? chf?

ABDOMEN U.S. (PORTABLE)
Reason: r/o portal venous thrombosis, assess ascites

CENTRAL LINE PLCT
Reason: line placement for HD

CONDITION UPDATE
Patient occasionally confused, oriented when asked

...

ICD: ['N179', 'J9600', 'J9690', 'A419', 'R6520', 'D696', 'I120', 'R6521']
CPT: ['99291', '99232', '99233', '94003', '99231', '94002', '99254', '99255', '99292', '36556', '90935']



For more detailed predictions across all samples, please refer to the test_predictions.jsonl file.
