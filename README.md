# 📝 Plagiarism Prevention Assistant

***Plagiarism Prevention Assistant*** è un progetto sviluppato per l'esame di Ingegneria della Conoscenza del corso di Informatica dell'Università degli Studi di Bari.  
Il suo scopo principale è quello di rilevare e prevenire il plagio accademico attraverso un sistema intelligente che non solo identifica il plagio, ma **educa** gli studenti su come citare correttamente le fonti.

---

## 🧠 Argomenti trattati

Nell'implementazione del progetto sono stati trattati i seguenti argomenti:

* **📘 Apprendimento Supervisionato**: il modello impara dal dataset `plagiarism_dataset.csv` fornito in input e viene addestrato per classificare diversi tipi e livelli di plagio;
* **📗 Apprendimento Supervisionato con Iperparametri**: il modello viene affinato per migliorare l'accuratezza delle sue previsioni attraverso tecniche di ottimizzazione;
* **📕 Sistema Esperto**: tramite una base di conoscenza e un modello inferenziale in Prolog, viene creato un *Knowledge Base System* che classifica il tipo di plagio, valuta la gravità e fornisce raccomandazioni personalizzate;
* **📙 Knowledge Graph**: rappresentazione delle relazioni tra tipi di plagio, discipline accademiche e regole di citazione attraverso un grafo di conoscenza.

---

## 🔍 Struttura del repository

Il repository contiene:

* `apprendimento_supervisionato`: file utilizzati per esplorare il dataset e visualizzare i risultati degli algoritmi di apprendimento supervisionato applicati al modello ***Plagiarism Prevention Assistant***;
* `dataset`: dataset utilizzato dal modello;
* `documentazione`: documentazione del progetto;
* `img`: grafici inseriti nella documentazione;
* `sistema_esperto`: Knowledge Base System relativo all'argomento preso in esame;
* `knowledge_graph`: rappresentazione grafica e logica del dominio del plagio accademico;
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

Prima di eseguire il progetto è necessario installare le dipendenze richieste (si consiglia di creare prima un ambiente virtuale e di attivarlo: <https://aulab.it/guide-avanzate/come-creare-un-virtual-environment-in-python>):

```bash
pip install -r requirements.txt
```

### 📍 Apprendimento Supervisionato

Spostandoci nella cartella `apprendimento_supervisionato` mediante il comando:

```bash
cd apprendimento_supervisionato
```

è possibile eseguire nell'ordine i file `preprocessing_simple.py` e `train_val_robust.py` per eseguire rispettivamente le fasi di *Preprocessing* e *Training and Evaluation* che rappresentano le prime tipiche fasi di un progetto di Machine Learning.  
Il comando da digitare è il seguente:

```bash
python nome_del_file.py
```

sostituendo **nome\_del\_file.py** con il file che si vuole eseguire (ad esempio: `python preprocessing.py`).

❗️Se si vogliono visualizzare direttamente le informazioni iniziali del dataset, il risultato della fase di preprocessing e le valutazioni dell'addestramento del modello, si consiglia di eseguire il file `train_val.py`; se, invece, si vogliono visualizzare dettagliatamente i risultati di ogni fase (compresi i grafici presenti nella documentazione e nella cartella `img`) si consiglia di eseguire separatamente ogni file nell'ordine descritto sopra.

### 📍 Apprendimento Supervisionato con Iperparametri

Per la fase di tuning, si è deciso di migliorare i modelli *Random Forest*, *SVM* e *KNN*.  
Per visualizzare i risultati dell'ottimizzazione di ogni modello, è possibile eseguire i file `optimized_random_forest.py`, `optimized_svm.py` e `optimized_knn.py` digitando lo stesso comando descritto precedentemente.

### 📍 Sistema Esperto

Per eseguire il Knowledge Base System è necessario installare l'ambiente di sviluppo [SWI-Prolog](https://www.swi-prolog.org/download/devel) (❗️spuntare l'aggiunta alla variabile *path*).  
Successivamente, è necessario navigare all'interno della cartella `sistema_esperto` con il seguente comando (se ci si trova nella cartella principale `ICON-Plagiarism_Prevention`):

```bash
cd sistema_esperto
```

oppure (se ci si trova nella cartella `apprendimento_supervisionato`):

```bash
cd ../sistema_esperto
```

e digitare il comando:

```bash
python expert_system_plagiarism.py
```

per lanciare l'interfaccia utente realizzata per il sistema esperto.

### 📍 Knowledge Graph

Per visualizzare il grafo di conoscenza del dominio del plagio accademico, navigare nella cartella `knowledge_graph`:

```bash
cd knowledge_graph
```

ed eseguire:

```bash
python visualize_kg.py
```

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
* Documentazione completa disponibile nella cartella `documentazione`

## About

Progetto per l'esame di Ingegneria della Conoscenza (A.A. 2025-2026)