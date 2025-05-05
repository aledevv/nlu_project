

# References
- configura BERT per fine-tuning [qui](https://github.com/HalflingWizard/FA-Intent-Classification-and-Slot-Filling/blob/main/model/joint_model.py) e soprattutto [quii](https://github.com/monologg/JointBERT/blob/master/main.py)


# 🔄 Migration from LSTM to BERT for Joint Intent Classification and Slot Filling

## 🚧 Key Architectural Changes

This project originally used an LSTM-based model (`ModelIAS`) to jointly perform intent classification and slot filling on the ATIS dataset. We replaced this architecture with a BERT-based model fine-tuned in a multi-task learning setup.

Here are the main modifications introduced:

### 1. ✅ Replacing LSTM with Pretrained BERT
- We switched from a trainable embedding + LSTM encoder to a pretrained `bert-base-uncased` model from HuggingFace.
- The BERT model provides token-level (`last_hidden_state`) and sequence-level (`pooler_output`) representations used for slot and intent predictions respectively.

### 2. ✅ Handling Sub-tokenization for Slot Labels
- BERT tokenizes words into sub-word units. Since the original slot labels are defined at the word level, we introduced a `slot_label_mask` to:
  - Assign the label only to the **first sub-token** of each word.
  - Mask out other sub-tokens and special tokens (`[CLS]`, `[SEP]`) during loss computation.

### 3. ✅ Custom Dataset and Dataloader
- We replaced the original dataset logic with a BERT-compatible `Dataset` class using `BertTokenizerFast`.
- A new `collate_fn` was created to produce `input_ids`, `attention_mask`, `token_type_ids`, and `slot_label_mask`.

### 4. ✅ Loss Computation Adjustments
- We continued using CrossEntropy loss, but ensured:
  - Intent loss is computed from `pooler_output`.
  - Slot loss is computed **only** over masked positions (`slot_label_mask == 1`).

### 5. ✅ Model Initialization and Dropout
- The `dropout_prob` is now configurable and applied to both intent and slot heads.
- The model is wrapped in a factory function for clean instantiation across multiple runs.

---

## 📊 Evaluation Strategy: Grid Search

To identify optimal hyperparameters, we implemented a grid search over:

- Learning rate: `[1e-5, 3e-5, 5e-5, 1e-4]`
- Dropout rate: `[0.1, 0.2, 0.3, 0.4]`

Each combination was tested under **three different batch size setups**:

| Experiment | Train Batch Size | Eval Batch Size |
|------------|------------------|-----------------|
| 1          | 128              | 64              |
| 2          | 64               | 32              |
| 3          | 32               | 16              |

Each configuration was trained for multiple runs (2, otherwise it would take too much time) to reduce randomness, and we logged:

- Mean Slot F1 Score
- Mean Intent Classification Accuracy

---

## 📈 Results Visualization

After each grid search, we produced a heatmap plotting `Slot F1` scores with:
- **Learning rate** on the x-axis
- **Dropout rate** on the y-axis

Each heatmap was saved as: experiments/grid_f1_train{train_batch}_eval{eval_batch}.png

This allowed visual comparison of the effect of hyperparameters under different training settings.

---