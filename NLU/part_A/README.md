

3 run per ogni esperimento.
Per ognuna salvo i pesi del modello migliore
salvo inoltre i risultati delle run per ogni esperimento e poi, in un log globale, tutti gli esperimenti.

Analisi
Considerando come primo fattore la f1 (che è un po' più bilanciata in teoria come metrica, rispetto all'accuracy -> della dev, no training):

## In generale:
I valori stanno tutti intorno 0.92-0.94 quindi le variazioni di f1 e accuracy non cambiano sensibilmente, a patto che non settiamo iperparametri con valori molto alti/bassi (in quel caso si vede già di più). 

- vanilla -> f1: 0.92

## Con le modifiche:
- solo dropout vine circa 0.92 che è il peggiore (sul summary c'è scritto 0.7 ma c'è stato un training un po' sfigato)
- solo bidirectional passa a poco più di 0.93 per cui migliora

## Con tutti e 2:
- con ambe le modifiche le performance passano in media a 0.94 (media su 3 runs) -> quindi meglio usarle tutte e 2 insieme (max performance raggiunta, sempre basandosi sull'eval)

Ho fatto poi un po' di prove cambiando gli iperparametri
- con entrambe le tecniche
    ### dropout
    - se aumento il dropout (0.3) le performance sono più o meno simili
    - con un dropout molto più alto (0.8) sia f1 che acc scendono (0.89 e 0.92)

    ### layers
    - se impostiamo 2 layers non cambia molto. Con 5 (accuracy aumenta ma f1 si abassa, quindi forse ha overfittato) e 10 (fa schifo, 0.79 di f1 e 0.95 di acc, ma non è veritiera, si vede anche da grafico che il training non è andato molto bene, va giù troppo veloce)

    ### emb size e hid size
    - aumentando alternativamente uno dei 2 le performance aumentano un pochino, ma ci sta dato che aumentiamo la complessità degli emebddings -> più dettagli e rappresentazione, o abbiamo una rete più complessa che quindi analogamente dovrebbe astrarre meglio
    - aumentando entrambi (hidden size 300 e emb size 400) non si vedono particolari miglioramenti, siamo sempre intorno a 0.94 di f1 e 0.95 di acc (sono leggermente più alti in media ma non significativamente, anche perchè l'architettura non è aumentata tanto)

    ### batch size
    - con un batch size più basso le performance aumentino (quasi 0.945)
    - inversamente, un batch size più alto tipo 256 train e 128 eval degrada un pochino le prestazioni (0.93 circa)

    ### learning rate
    - se abbasso learning rate sotto 10^-4 -> quindi circa 10^-5 performa un po' peggio
    - se lo passo a 5e-4 (5*10^-4) anche li no buono passa a 0.92/0.93
    - (se also il lr rimuovendo una delle due modifiche segue le regole relative alle modifiche, dropout da solo da un po' schifo)
    - Valori estremi:
        - alto lr (0.01) -> performance un po' diminuite -> 0.93
        - basso lr (10^-6) -> si è impinatato dopo poche epoche (15 circa) e non ha imparato (= impara lentamente e la patience ha troncato il training)

# TODO
- ✅ fare grafici per fare un interpretazione
- ✅ impostare diversi config e provarli in massa e vedere come cambia il risultato