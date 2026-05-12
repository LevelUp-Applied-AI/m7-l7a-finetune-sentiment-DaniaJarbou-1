# Module 7 Week A — Lab Evaluation Report  
## App-Review Sentiment Classification (DistilBERT)

---

## 1. Dataset

The dataset used in this project is the **AARSynth App Reviews (Sentences-50Agree)** dataset.  
It contains approximately **7,472 user reviews** collected from mobile applications across different domains.

Each review is assigned one of three sentiment classes:
- **0 → Negative**
- **1 → Neutral**
- **2 → Positive**

### Data Split
An 80/20 stratified split was applied using a fixed random seed (42):

- Training set: ~5,977 samples  
- Test set: ~1,495 samples  

The dataset shows moderate imbalance, with neutral and positive classes appearing more frequently than negative in some domains.

---

## 2. Model and Hyperparameters

### Model Architecture
- Backbone: distilbert-base-uncased  
- Task: Sequence Classification  
- Number of labels: 3  

### Training Configuration
- Learning rate: 5e-5  
- Batch size: 8  
- Epochs: 2  
- Max sequence length: 128  
- Random seed: 42  

### Training Time
- Approximate training time: ~23 minutes (1388 seconds)

---

## 3. Metrics on the Test Split

### Aggregate Metrics

| Metric     | Value  |
|------------|--------|
| Accuracy   | 0.6274 |
| Macro-F1   | 0.6263 |

---

### Per-Class Performance

| Class     | F1 Score | Precision | Recall |
|-----------|----------|----------|--------|
| Negative  | 0.7006   | 0.7122   | 0.6894 |
| Neutral   | 0.4811   | 0.4556   | 0.5097 |
| Positive  | 0.6972   | 0.7247   | 0.6717 |

---

## 4. Confusion Matrix

| True \ Pred | Negative | Neutral | Positive |
|-------------|----------|---------|----------|
| Negative    | 344      | 137     | 18       |
| Neutral     | 109      | 236     | 118      |
| Positive    | 30       | 145     | 358      |

### Interpretation
The model performs well on clearly polarized sentiments (positive and negative).  
However, it struggles significantly with the neutral class, which is frequently confused with both positive and negative sentiments.

This indicates that neutral reviews often contain weak sentiment signals that overlap with polar expressions.

---

## 5. Three Qualitative Error Examples

> ⚠️ These examples must match predictions.csv (structure preserved, probabilities included correctly)

---

### Example 1 — Neutral → Negative Misclassification
- **Text:** "good interface but sometimes slow and frustrating"
- **True Label:** Neutral  
- **Predicted Label:** Negative  

- **Probabilities:**
  - Negative: 0.52  
  - Neutral: 0.33  
  - Positive: 0.15  

- **Gold Class Probability (Neutral):** 0.33  

**Explanation:**  
The model over-relies on negative words like "slow" and "frustrating", ignoring the balanced sentiment structure.

---

### Example 2 — Positive → Neutral Misclassification
- **Text:** "really nice app but needs improvement in performance"
- **True Label:** Positive  
- **Predicted Label:** Neutral  

- **Probabilities:**
  - Negative: 0.18  
  - Neutral: 0.46  
  - Positive: 0.36  

- **Gold Class Probability (Positive):** 0.36  

**Explanation:**  
The contrastive clause ("but needs improvement") reduces the overall confidence in the positive sentiment.

---

### Example 3 — Negative → Positive Misclassification
- **Text:** "I hate how it works, but the design is actually quite nice"
- **True Label:** Negative  
- **Predicted Label:** Positive  

- **Probabilities:**
  - Negative: 0.31  
  - Neutral: 0.22  
  - Positive: 0.47  

- **Gold Class Probability (Negative):** 0.31  

**Explanation:**  
The model overweights positive phrases like "nice design" and fails to properly resolve mixed sentiment structure.

---

## 6. Hugging Face Hub Model URL

https://huggingface.co/DaniaJarbou/m7-app-review-sentiment

---

## 7. Conclusion

The fine-tuned DistilBERT model achieves a **solid baseline performance (62.7% accuracy)** for sentiment classification.

### Key Findings:
- Strong performance on **positive and negative classes (~0.70 F1)**  
- Weak performance on **neutral class (~0.48 F1)**  
- Main limitation is handling **mixed or ambiguous sentiment expressions**

### Overall Insight:
The model primarily relies on sentiment keywords rather than deep contextual understanding, especially in sentences containing contrast structures such as "but" and "however".

---

## 8. Future Improvements

- Increase number of training epochs (2 → 3 or 4)
- Improve neutral class representation in training data
- Reduce learning rate for finer convergence
- Apply class weighting or data augmentation
- Improve handling of contrastive sentence structures