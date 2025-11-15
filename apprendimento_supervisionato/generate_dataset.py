"""
Generatore di Dataset Sintetico per Plagiarism Detection
Questo script genera un dataset realistico con diversi tipi di plagio:
- No Plagiarism: testi originali
- Literal Plagiarism: copia letterale
- Mosaic Plagiarism: frasi da fonti diverse
- Paraphrasing: riformulazione insufficiente
- Idea Plagiarism: stesse idee senza citazione

Autore: [Raffaella Nitti]
Matricola: [796132]
"""

import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import os

print("="*80)
print("🔧 GENERAZIONE DATASET PLAGIARISM")
print("="*80)

# Seed per riproducibilità
np.random.seed(42)
random.seed(42)

# ==================== TESTI BASE ====================
# Questi sono "testi fonte" da cui generare plagio

SOURCE_TEXTS = [
    "Artificial intelligence is transforming modern society through machine learning algorithms that can process vast amounts of data.",
    "Climate change represents one of the most significant challenges facing humanity in the 21st century.",
    "The human brain contains approximately 86 billion neurons that communicate through electrical and chemical signals.",
    "Quantum computing promises to revolutionize cryptography and computational chemistry through quantum superposition.",
    "The Renaissance period marked a cultural rebirth in Europe characterized by advances in art, science, and philosophy.",
    "Photosynthesis is the process by which plants convert light energy into chemical energy stored in glucose molecules.",
    "Global economic inequality has increased dramatically over the past four decades due to various structural factors.",
    "The discovery of penicillin by Alexander Fleming revolutionized modern medicine and saved millions of lives.",
    "Social media platforms have fundamentally altered how people communicate and share information globally.",
    "DNA molecules contain genetic instructions for the development and functioning of all living organisms.",
    "The Industrial Revolution transformed societies from agrarian economies to industrial manufacturing systems.",
    "Renewable energy sources such as solar and wind power are essential for sustainable development.",
    "The human genome consists of approximately three billion base pairs of DNA sequences.",
    "Ancient Greek philosophy laid the foundations for Western scientific and philosophical thought.",
    "The theory of relativity changed our understanding of space, time, and gravity fundamentally.",
    "Biodiversity loss threatens ecosystem stability and human well-being across the planet.",
    "The invention of the printing press democratized access to knowledge and accelerated cultural exchange.",
    "Neural networks are computational models inspired by biological neural systems in animal brains.",
    "Ocean acidification caused by carbon dioxide absorption poses serious threats to marine ecosystems.",
    "The scientific method provides a systematic approach to investigating phenomena and acquiring knowledge.",
]

# ==================== FUNZIONI DI GENERAZIONE PLAGIO ====================

def generate_no_plagiarism():
    """Genera testo originale (nessun plagio)"""
    templates = [
        "Research has shown that {} can lead to {}.",
        "Studies indicate that {} is increasingly important for {}.",
        "Experts suggest that {} plays a crucial role in {}.",
        "Recent findings demonstrate that {} contributes significantly to {}.",
        "Analysis reveals that {} has profound implications for {}.",
    ]
    
    topics = ["technology", "education", "healthcare", "environment", "economics", 
              "innovation", "sustainability", "development", "research", "society"]
    outcomes = ["improved outcomes", "better understanding", "enhanced performance", 
                "positive changes", "significant progress", "new opportunities"]
    
    template = random.choice(templates)
    topic = random.choice(topics)
    outcome = random.choice(outcomes)
    
    return template.format(topic, outcome)


def generate_literal_plagiarism():
    """Genera plagio letterale (copia esatta)"""
    source = random.choice(SOURCE_TEXTS)
    # Copia letterale con minime modifiche (punteggiatura, capitalizzazione)
    if random.random() > 0.5:
        return source
    else:
        # Piccole variazioni
        return source.replace(".", " and this is significant.")


def generate_mosaic_plagiarism():
    """Genera plagio mosaico (frasi da più fonti)"""
    # Prendi parti da 2-3 fonti diverse
    num_sources = random.randint(2, 3)
    selected = random.sample(SOURCE_TEXTS, num_sources)
    
    # Dividi ogni testo in parti e ricombina
    parts = []
    for text in selected:
        words = text.split()
        mid = len(words) // 2
        parts.append(" ".join(words[:mid]))
    
    return " ".join(parts) + "."


def generate_paraphrasing():
    """Genera parafrasi insufficiente"""
    source = random.choice(SOURCE_TEXTS)
    
    # Sostituzioni semplici (parafrasi scarsa)
    replacements = {
        "is": "represents",
        "are": "constitute",
        "through": "via",
        "can": "may",
        "has": "possesses",
        "the": "a",
        "most": "very",
        "significant": "important",
        "modern": "contemporary",
        "various": "multiple",
        "fundamental": "basic",
    }
    
    result = source
    for old, new in replacements.items():
        if old in result:
            result = result.replace(old, new, 1)
    
    return result


def generate_idea_plagiarism():
    """Genera plagio di idee (concetto copiato, parole diverse)"""
    source = random.choice(SOURCE_TEXTS)
    
    # Cambia struttura ma mantieni l'idea
    if "is" in source or "are" in source:
        words = source.split()
        # Ristruttura la frase
        return f"It can be observed that {' '.join(words[2:])}"
    else:
        return f"Research indicates that {source.lower()}"


# ==================== GENERAZIONE DATASET ====================

def generate_dataset(n_samples=1000):
    """
    Genera dataset bilanciato con diversi tipi di plagio
    
    Parameters:
    -----------
    n_samples : int
        Numero totale di campioni da generare
    """
    
    print(f"\n📊 Generazione di {n_samples} campioni...")
    
    data = []
    labels = ['no_plagiarism', 'literal', 'mosaic', 'paraphrasing', 'idea']
    samples_per_class = n_samples // len(labels)
    
    # Genera campioni per ogni classe
    for label in tqdm(labels, desc="Generazione classi"):
        for _ in range(samples_per_class):
            if label == 'no_plagiarism':
                text = generate_no_plagiarism()
            elif label == 'literal':
                text = generate_literal_plagiarism()
            elif label == 'mosaic':
                text = generate_mosaic_plagiarism()
            elif label == 'paraphrasing':
                text = generate_paraphrasing()
            elif label == 'idea':
                text = generate_idea_plagiarism()
            
            # Aggiungi metadati utili
            data.append({
                'text': text,
                'label': label,
                'text_length': len(text.split()),
                'char_length': len(text)
            })
    
    # Crea DataFrame
    df = pd.DataFrame(data)
    
    # Shuffle dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df


# ==================== ESECUZIONE ====================

if __name__ == "__main__":
    
    # Genera dataset
    print("\n🎲 Generazione dataset in corso...")
    df = generate_dataset(n_samples=1000)
    
    print(f"\n✅ Dataset generato!")
    print(f"📊 Dimensioni: {df.shape[0]} campioni, {df.shape[1]} colonne")
    
    # Statistiche
    print("\n📈 Distribuzione classi:")
    print(df['label'].value_counts())
    
    print("\n📏 Statistiche lunghezza testi:")
    print(df.groupby('label')['text_length'].describe())
    
    # Salva dataset
    output_path = '../dataset/plagiarism_dataset.csv'
    
    # Crea cartella se non esiste
    os.makedirs('../dataset', exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"\n💾 Dataset salvato in: {output_path}")
    
    # Mostra alcuni esempi
    print("\n" + "="*80)
    print("📝 ESEMPI DI TESTI GENERATI")
    print("="*80)
    
    for label in df['label'].unique():
        print(f"\n🔸 Classe: {label.upper()}")
        example = df[df['label'] == label].iloc[0]['text']
        print(f"   Testo: {example[:150]}...")
    
    print("\n" + "="*80)
    print("✅ GENERAZIONE COMPLETATA!")
    print("="*80)
    print("\n🎯 Prossimo step: eseguire 'preprocessing.py'")
    print("="*80)