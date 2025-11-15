"""
Preprocessing Semplificato (senza NLTK)
Questo script preprocessa il dataset senza dipendenze complicate

Autore: [Raffaella Nitti]
Matricola: [796132]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("📝 PLAGIARISM PREVENTION ASSISTANT - PREPROCESSING SEMPLIFICATO")
print("="*80)

# ==================== CARICAMENTO DATASET ====================
print("\n📂 Caricamento dataset...")

try:
    df = pd.read_csv('../dataset/plagiarism_dataset.csv')
    print("✅ Dataset caricato con successo!")
    print(f"📊 Dimensioni dataset: {df.shape[0]} righe, {df.shape[1]} colonne")
except FileNotFoundError:
    print("❌ ERRORE: File dataset non trovato!")
    print("Esegui prima 'python3 generate_dataset.py' per creare il dataset")
    exit(1)

# ==================== ESPLORAZIONE DATI ====================
print("\n" + "="*80)
print("🔍 ESPLORAZIONE INIZIALE DEL DATASET")
print("="*80)

print("\n📋 Prime 5 righe del dataset:")
print(df.head())

print("\n📊 Informazioni sulle colonne:")
print(df.info())

print("\n📈 Statistiche descrittive:")
print(df.describe())

print("\n🔢 Valori mancanti per colonna:")
print(df.isnull().sum())

print("\n📊 Distribuzione delle classi:")
print(df['label'].value_counts())

# Visualizzazione distribuzione classi
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='label', palette='viridis')
plt.title('Distribuzione delle Classi di Plagio', fontsize=16, fontweight='bold')
plt.xlabel('Tipo di Plagio', fontsize=12)
plt.ylabel('Numero di Istanze', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('../img/distribuzione_classi.png', dpi=300, bbox_inches='tight')
print("\n✅ Grafico distribuzione classi salvato in 'img/distribuzione_classi.png'")
plt.close()

# ==================== PREPROCESSING TESTO SEMPLIFICATO ====================
print("\n" + "="*80)
print("🧹 PREPROCESSING DEL TESTO (metodo semplificato)")
print("="*80)

# Stopwords semplici in inglese (lista manuale)
STOPWORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 
    'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 
    'will', 'with', 'this', 'but', 'they', 'have', 'had', 'what', 'when', 
    'where', 'who', 'which', 'why', 'how', 'can', 'or', 'if', 'their', 'there'
}

def preprocess_text_simple(text):
    """
    Preprocessa il testo in modo semplificato:
    - Converte in minuscolo
    - Rimuove punteggiatura
    - Rimuove stopwords
    """
    if pd.isna(text):
        return ""
    
    # Converti in minuscolo
    text = text.lower()
    
    # Rimuovi tutto tranne lettere e spazi
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Rimuovi spazi multipli
    text = re.sub(r'\s+', ' ', text)
    
    # Tokenizza e rimuovi stopwords
    words = text.split()
    words = [w for w in words if w not in STOPWORDS and len(w) > 2]
    
    return ' '.join(words)

print("\n⚙️ Applicazione preprocessing al testo...")
df['text_clean'] = df['text'].apply(preprocess_text_simple)
print("✅ Preprocessing completato!")

print("\n📝 Esempio di testo preprocessato:")
print(f"\n🔸 Originale:\n{df['text'].iloc[0][:200]}...")
print(f"\n🔸 Preprocessato:\n{df['text_clean'].iloc[0][:200]}...")

# Statistiche lunghezza testi
df['text_length_clean'] = df['text_clean'].apply(lambda x: len(x.split()))
print(f"\n📏 Lunghezza media testo preprocessato (parole): {df['text_length_clean'].mean():.2f}")
print(f"📏 Lunghezza min: {df['text_length_clean'].min()}")
print(f"📏 Lunghezza max: {df['text_length_clean'].max()}")

# Visualizzazione distribuzione lunghezza testi
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(df['text_length_clean'], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribuzione Lunghezza Testi', fontsize=14, fontweight='bold')
plt.xlabel('Numero di Parole', fontsize=12)
plt.ylabel('Frequenza', fontsize=12)

plt.subplot(1, 2, 2)
sns.boxplot(data=df, x='label', y='text_length_clean', palette='Set2')
plt.title('Lunghezza Testi per Classe', fontsize=14, fontweight='bold')
plt.xlabel('Tipo di Plagio', fontsize=12)
plt.ylabel('Numero di Parole', fontsize=12)
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('../img/distribuzione_lunghezza_testi.png', dpi=300, bbox_inches='tight')
print("\n✅ Grafico lunghezza testi salvato in 'img/distribuzione_lunghezza_testi.png'")
plt.close()

# ==================== FEATURE EXTRACTION ====================
print("\n" + "="*80)
print("🔧 ESTRAZIONE FEATURES (TF-IDF)")
print("="*80)

# Inizializza TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=500, min_df=2, max_df=0.8, ngram_range=(1, 2))

print("\n⚙️ Calcolo TF-IDF features...")
X_tfidf = tfidf.fit_transform(df['text_clean'])
print(f"✅ Features estratte: {X_tfidf.shape[1]} features TF-IDF")

# Le top 20 parole più importanti
feature_names = tfidf.get_feature_names_out()
tfidf_scores = X_tfidf.toarray().sum(axis=0)
top_indices = tfidf_scores.argsort()[-20:][::-1]
top_words = [feature_names[i] for i in top_indices]

print("\n🔝 Top 20 features più rilevanti:")
for i, word in enumerate(top_words, 1):
    print(f"{i}. {word}")

# Visualizzazione top words
plt.figure(figsize=(12, 6))
plt.barh(range(len(top_words)), [tfidf_scores[feature_names.tolist().index(w)] for w in top_words], 
         color='coral', edgecolor='black')
plt.yticks(range(len(top_words)), top_words)
plt.xlabel('Score TF-IDF', fontsize=12)
plt.title('Top 20 Features più Rilevanti (TF-IDF)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../img/top_words_tfidf.png', dpi=300, bbox_inches='tight')
print("\n✅ Grafico top features salvato in 'img/top_words_tfidf.png'")
plt.close()

# ==================== TRAIN-TEST SPLIT ====================
print("\n" + "="*80)
print("✂️ SPLIT TRAIN-TEST")
print("="*80)

X = X_tfidf
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✅ Dataset splittato!")
print(f"📊 Training set: {X_train.shape[0]} campioni")
print(f"📊 Test set: {X_test.shape[0]} campioni")
print(f"📊 Features: {X_train.shape[1]}")

print("\n📈 Distribuzione classi nel training set:")
print(y_train.value_counts())
print("\n📈 Distribuzione classi nel test set:")
print(y_test.value_counts())

# ==================== SALVATAGGIO ====================
print("\n" + "="*80)
print("💾 SALVATAGGIO DATI PREPROCESSATI")
print("="*80)

# Salva il dataframe preprocessato
df.to_csv('../dataset/plagiarism_dataset_preprocessed.csv', index=False)
print("✅ Dataset preprocessato salvato in 'dataset/plagiarism_dataset_preprocessed.csv'")

# Salva le features e i label
np.save('../dataset/X_train.npy', X_train.toarray())
np.save('../dataset/X_test.npy', X_test.toarray())
np.save('../dataset/y_train.npy', y_train.values)
np.save('../dataset/y_test.npy', y_test.values)
print("✅ Features e labels salvati in formato .npy")

# Salva il vectorizer per uso futuro
import joblib
joblib.dump(tfidf, '../dataset/tfidf_vectorizer.pkl')
print("✅ TF-IDF Vectorizer salvato in 'dataset/tfidf_vectorizer.pkl'")

print("\n" + "="*80)
print("✅ PREPROCESSING COMPLETATO CON SUCCESSO!")
print("="*80)
print("\n📁 File generati:")
print("   - dataset/plagiarism_dataset_preprocessed.csv")
print("   - dataset/X_train.npy, X_test.npy")
print("   - dataset/y_train.npy, y_test.npy")
print("   - dataset/tfidf_vectorizer.pkl")
print("   - img/distribuzione_classi.png")
print("   - img/distribuzione_lunghezza_testi.png")
print("   - img/top_words_tfidf.png")
print("\n🎯 Prossimo step: eseguire 'python3 train_val_robust.py' per addestrare i modelli!")
print("="*80)