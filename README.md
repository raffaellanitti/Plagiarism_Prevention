# 📝 Plagiarism Prevention Assistant

***Plagiarism Prevention Assistant*** è un progetto sviluppato per l'esame di Ingegneria della Conoscenza del corso di Informatica dell'Università degli Studi di Bari.  
Il suo scopo principale è quello di rilevare e prevenire il plagio accademico attraverso un sistema intelligente che non solo identifica il plagio, ma **educa** gli studenti su come citare correttamente le fonti.

---

## 🧠 Argomenti trattati

Nell'implementazione del progetto sono stati trattati i seguenti argomenti:

* **📘 Apprendimento Supervisionato**: il modello impara dal dataset `plagiarism_dataset.csv` generato sinteticamente e viene addestrato per classificare diversi tipi e livelli di plagio (letterale, mosaico, parafrasi insufficiente, idee, nessuno);
* **📗 Apprendimento Supervisionato con Iperparametri**: il modello viene affinato per migliorare l'accuratezza delle sue previsioni attraverso tecniche di ottimizzazione (GridSearchCV su Random Forest, SVM, KNN);
* **📕 Sistema Esperto**: tramite una base di conoscenza e un modello inferenziale in Prolog, viene creato un *Knowledge Base System* che classifica il tipo di plagio, valuta la gravità contestuale (considerando disciplina e livello di studio) e fornisce raccomandazioni personalizzate;
* **📙 Sistema Ibrido ML + KB**: integrazione di Machine Learning e reasoning simbolico per decisione più accurate, dove la Knowledge Base valida e corregge le predizioni del modello ML quando necessario.

---

## 🔍 Struttura del repository

Il repository contiene:

* `apprendimento_supervisionato`: file utilizzati per esplorare il dataset e visualizzare i risultati degli algoritmi di apprendimento supervisionato applicati al modello ***Plagiarism Prevention Assistant***;
* `dataset`: dataset utilizzato dal modello;
* `documentazione`: documentazione del progetto;
* `img`: grafici inseriti nella documentazione;
* `sistema_esperto`: Knowledge Base System in Prolog con interfaccia grafica;
* `integration`: Sistema Ibrido che integra Machine Learing e Knowledge Base;
* `requirements.txt`: file con l'elenco di tutte le dipendenze necessarie.

---

## ▶️ Esecuzione

Innanzitutto, è necessario aprire il terminale e clonare il repository con il seguente comando:

```bash
git clone https://github.com/raffaellanitti/ICON-Plagiarism_Prevention.git
```

e navigare all'interno della cartella principale:

```bash
cd ICON-Plagiarism_Prevention
```

Prima di eseguire il progetto è necessario installare le dipendenze richieste (si consiglia di creare prima un ambiente virtuale e di attivarlo):

### Creazione ambiente virtuale:**

**Mac/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate 
```

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate 
```

### 📦Installazione dipendenze:**

```bash
pip install -r requirements.txt
```

### Installazione SWI-Prolog:

SWI-Prolog è necessario per il sistema esperto.

* Mac: `brew install swi-prolog`
* Windows/Linux: Download Sito Ufficiale

### 📍 Apprendimento Supervisionato

Spostandoci nella cartella `apprendimento_supervisionato` mediante il comando:

```bash
cd apprendimento_supervisionato
```

è possibile eseguire nell'ordine i file `preprocessing.py` e `train_val.py` per eseguire rispettivamente le fasi di *Preprocessing* e *Training and Evaluation* che rappresentano le prime tipiche fasi di un progetto di Machine Learning.  
Il comando da digitare è il seguente:

**Mac/Linux:**

```bash
python3 nome_del_file.py
```
**Windows:**

```bash
python nome_del_file.py
```

sostituendo **nome\_del\_file.py** con il file che si vuole eseguire (ad esempio: `python preprocessing.py`).

❗️Se si vogliono visualizzare direttamente le informazioni iniziali del dataset, il risultato della fase di preprocessing e le valutazioni dell'addestramento del modello, si consiglia di eseguire il file `train_val.py`; se, invece, si vogliono visualizzare dettagliatamente i risultati di ogni fase (compresi i grafici presenti nella documentazione e nella cartella `img`) si consiglia di eseguire separatamente ogni file nell'ordine descritto sopra.

### 📍 Apprendimento Supervisionato con Iperparametri

Per ottimizzare i tre modelli migliori:

**Mac/Linux:**

```bash
# Random Forest (richiede ~5-10 minuti)
python3 optimized_random_forest.py

# SVM (richiede ~5-10 minuti)
python3 optimized_svm.py

# KNN (richiede ~3-5 minuti)
python3 optimized_knn.py
```

**Windows:**

```bash
# Random Forest (richiede ~5-10 minuti)
python optimized_random_forest.py

# SVM (richiede ~5-10 minuti)
python optimized_svm.py

# KNN (richiede ~3-5 minuti)
python optimized_knn.py
```

Ogni script:
* Esegue GridSearchCV con 5-fold CV
* Confronta baseline vs ottimizzato
* Genera grafici comparativi
* Salva il modello ottimizzato 

### 📍 Sistema Esperto

**Prerequisiti:**
  * SWI-Prolog installato e aggiunto al PATH

**Esecuzione:**
Navigare nella cartella del sistema esperto: 

```bash
cd sistema_esperto
```

Eseguire il sistema esperto:

**Mac/Linux:**

```bash
python3 expert_system_plagiarism.py
```

**Windows:**

```bash
python expert_system_plagiarism.py
```

Si aprirà un'interfaccia grafica dove è possibile:
* Impostare parametri di input (similarità, percentuale, disciplina, ecc..)
* Analizzare il testo tramite reasoning Prolog
* Visualizzare tipo di plagio, gravità e raccomandazioni

La Knowledge Base include:
* 50+ regole Prolog
* Reasoning multi-livello
* Valutazione contestuale per disciplina e livello studio
* Raccomandazioni personalizzate

### 📍 Sistema Ibrido ML + KB

Per testare il sistema ibrido che integra Machine Learning e Knowledge Base:

```bash
cd integration
```

Esegui il:

**Mac/Linux:**

```bash
# Demo base
python3 hybrid_system.py

# Test casi complessi
python3 test_hybrid_system.py
```

**Windows:**

```bash
# Demo base
python hybrid_system.py

# Test casi complessi
python test_hybrid_system.py
```

Il sistema ibrido combina le predizioni del modello ML con il reasoning simbolico della Knowledge Base per decisioni più accurate. In caso di disaccordo tra ML e KB, il sistema esperto prevale (approccio conservativo).

---

## 🚀 Sviluppi futuri

* Integrazione con editor di testo (plugin per Word/Google Docs)
* Supporto multilingua per rilevamento plagio
* Sistema di raccomandazione automatica di parafrasazioni
* Dashboard per docenti con statistiche di classe
* Estensione del sistema esperto con più regole disciplina-specific

---

## 👤 Autore

Realizzato da:

* **[Raffaella Nitti]**: matricola: [796132], email: [r.nitti25@studenti.uniba.it](mailto:r.nitti25@studenti.uniba.it)

---

## 📚 Riferimenti

* Dataset: [Kaggle Plagiarism Detection](https://www.kaggle.com/datasets/...)
* Tecnologie: Python 3.12, scikit-learn, SWI-Prolog, Tkinter
* Knowledge base: 50+ regole Prolog per reasoning complesso
* Documentazione completa disponibile nella cartella `documentazione`

---