# Named Entity Recognition (NER) using Bidirectional LSTM from Scratch

A clean, high-performance sequence labeling pipeline built to perform Named Entity Recognition (NER) on the CoNLL-2003 dataset. This repository demonstrates how to implement specialized token preprocessing, custom batch collating with dynamic padding, and  loss handling using both a **Bidirectional LSTM** and  **Simple Linear**  architecture in PyTorch.

Rather than relying on high-level abstraction wrappers, the preprocessing pipelines, vocabulary tokenization mapping, and tracking metrics in this repository are written natively to provide clear insight into sequence modeling mechanics.



## 📂 Project Structure

```text
├── data_utils.py          # custom text parsing and preprocessing 
├── LSTM_NER.py            # Bidirectional LSTM Implementation
├── simple_NER.py          # Simple Dense layer Implementation
└── README.md              # Repository overview and execution documentation
├── .gitignore             # Shields local datasets, IDE configurations, and model weights

```

## Inference Report for: "George Washington traveled to Paris last week" using the LSTM model

```
Word               | Predicted Class ID | Entity Tag
-----------------------------------------------------
George             | 1                  | B-PER
Washington         | 2                  | I-PER
traveled           | 0                  | O
to                 | 0                  | O
Paris              | 5                  | B-LOC
last               | 0                  | O
week               | 0                  | O
=====================================================

```


## Detailed Metrics of the LSTM model on the validation dataset 
the model was trained for 10 epochs with ADAM optimization , a hidden dimension of 128 and learning rate of 0.001
```
================ DETAILED MULTICLASS METRICS ================
Total True Entity Tokens in Val Set : 8603
Correctly Predicted Entity Tokens   : 6583
False Positives (Spurious Alarms)   : 143
-------------------------------------------------------------
Macro Precision : 0.9009  (When it finds an entity, how often is it right?)
Macro Recall    : 0.7537  (What % of the actual entities did it catch?)
Macro F1-Score  : 0.8178  (Overall structural performance balance)
