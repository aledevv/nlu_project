

# TODO
- configura BERT per fine-tuning [qui](https://github.com/HalflingWizard/FA-Intent-Classification-and-Slot-Filling/blob/main/model/joint_model.py) e soprattutto [quii](https://github.com/monologg/JointBERT/blob/master/main.py)
    - serve tokenizzare il dataset
        > [!caution] Tokenizers
        >  Il tokenizer di spaCy divide seguendo regole grammaticali e casi specificy (come can't = "ca"+"n't"), mentre quello di HuggingFace (e quindi di transformers) fa BPE (Byte-Pair Ecoding) quindi divide le parole in pezzetti più piccoli seguendo modelli probabilistici. Diciamo che HuggingFace tokenizza per rendere l'output più suitable per i modelli di deep learning, mentre spaCy segue di più un approccio orientato alle regole linguistiche.
    - esegui fit con tale dataset
- configura che iperparametri cambiare
- controlla problema di sub-tokenization
- Configura per eseguire fine-tuning su BERT-base e BERT-large
- vedi quando ci mette ad allenare -> di conseguenza vedi se fare multi run


# Note
- ho configurato il modello bert così che fosse compatibile con il train e eval loops usati di solito, per ci ho fatto ritornare i logits di intent e slots
- 

# model.py
Model Adaptation (model.py)

Replace LSTM with BERT: Instead of the LSTM-based encoder, we'll load a pre-trained BERT model from Hugging Face Transformers. We'll use BertForTokenClassification as it's suitable for token-level classification (slot filling). For intent, we can either add a linear layer on top of BERT's [CLS] token representation or use BertForSequenceClassification.
Sub-word Tokenization: BERT uses sub-word tokenization. This means words can be broken into smaller units. We need to handle this in both training and evaluation to align slot labels correctly.
Multi-task Head: We'll maintain separate output layers for intent classification and slot filling, allowing the model to perform both tasks.

# for data preparation
Data Preparation (utils.py and functions.py)

Tokenization: Use the BERT tokenizer (BertTokenizerFast) to tokenize the input utterances. This will handle sub-word tokenization.
Input IDs and Attention Masks: Create input_ids and attention_mask as required by BERT. input_ids are the numerical representations of the tokens, and attention_mask indicates which tokens should be attended to (not padding).
Slot Alignment: Crucially, when creating the slot_ids, you'll need to align the slot labels with the sub-words produced by the BERT tokenizer. This is the trickiest part. You might need to expand slot labels or use special tokens ([CLS], [SEP]) and padding appropriately.
Data Class Changes: Modify the IntentsAndSlots dataset class to return input_ids, attention_mask, intent_ids, and slot_ids.
Padding: Adjust the collate_fn to pad input_ids and attention_mask correctly.

3. Training and Evaluation (functions.py)

Model Initialization: Load the pre-trained BERT model in init_model. You'll need to specify the number of intent and slot labels.
Forward Pass: Modify the train_loop and eval_loop to pass input_ids, attention_mask, intent_ids, and slot_ids to the BERT model.
Loss Calculation: Calculate the loss for intent classification and slot filling separately and then combine them.
Evaluation: In eval_loop, adapt the slot filling evaluation to handle BERT's tokenization. You'll need to map the predicted sub-word tokens back to the original words or use a token-level evaluation that ignores sub-word pieces. Ensure you're still using conll.evaluate for the slot filling F1 score. For intent, keep the accuracy calculation.