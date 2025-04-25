

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