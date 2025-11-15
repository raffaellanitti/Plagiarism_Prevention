"""
Sistema Ibrido: Integrazione Machine Learning + Knowledge Base
Combina predizioni ML con reasoning simbolico per decisioni più accurate

Autore: [Nome Cognome]
Matricola: [XXXXXX]
"""

import numpy as np
import pandas as pd
import joblib
from pyswip import Prolog
import os
from sklearn.feature_extraction.text import TfidfVectorizer

class HybridPlagiarismSystem:
    """
    Sistema Ibrido che combina:
    - ML: predizione tipo plagio e similarità
    - KB: validazione, reasoning contestuale, raccomandazioni
    """
    
    def __init__(self):
        print("="*80)
        print("🤖 INIZIALIZZAZIONE SISTEMA IBRIDO ML+KB")
        print("="*80)
        
        # Carica modelli ML
        self.load_ml_models()
        
        # Inizializza Knowledge Base
        self.init_knowledge_base()
        
        print("\n✅ Sistema Ibrido pronto!")
        print("="*80)
        
    def load_ml_models(self):
        """Carica i modelli ML addestrati"""
        print("\n📂 Caricamento modelli ML...")
        
        try:
            # Carica il miglior modello (Random Forest ottimizzato)
            self.ml_model = joblib.load('../dataset/best_model.pkl')
            print("   ✅ Modello ML caricato: Random Forest")
            
            # Carica vectorizer TF-IDF
            self.vectorizer = joblib.load('../dataset/tfidf_vectorizer.pkl')
            print("   ✅ TF-IDF Vectorizer caricato")
            
            # Mappa label → tipo plagio
            self.label_map = {
                'literal': 'letterale',
                'mosaic': 'mosaico',
                'paraphrasing': 'parafrasi_insufficiente',
                'idea': 'idee',
                'no_plagiarism': 'nessuno'
            }
            
        except Exception as e:
            print(f"   ❌ Errore caricamento modelli: {e}")
            raise
            
    def init_knowledge_base(self):
        """Inizializza la Knowledge Base Prolog"""
        print("\n🧠 Inizializzazione Knowledge Base...")
        
        try:
            self.prolog = Prolog()
            kb_path = os.path.join('..', 'sistema_esperto', 'plagiarism_rules.pl')
            self.prolog.consult(kb_path)
            print("   ✅ Knowledge Base caricata")
        except Exception as e:
            print(f"   ❌ Errore caricamento KB: {e}")
            raise
            
    def preprocess_text(self, text):
        """Preprocessa il testo per l'analisi"""
        # Applica stesso preprocessing del training
        import re
        
        STOPWORDS = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 
            'to', 'was', 'will', 'with', 'this', 'but', 'they', 'have', 'had'
        }
        
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        words = text.split()
        words = [w for w in words if w not in STOPWORDS and len(w) > 2]
        
        return ' '.join(words)
        
    def ml_predict(self, text):
        """
        Predizione ML del tipo di plagio
        
        Returns:
            dict: {
                'tipo_plagio': str,
                'confidence': float,
                'probabilities': dict
            }
        """
        # Preprocessa
        text_clean = self.preprocess_text(text)
        
        # Vettorizza
        X = self.vectorizer.transform([text_clean])
        
        # Predizione
        prediction = self.ml_model.predict(X)[0]
        
        # Probabilità
        if hasattr(self.ml_model, 'predict_proba'):
            proba = self.ml_model.predict_proba(X)[0]
            confidence = proba.max()
            
            # Mappa classi → probabilità
            classes = self.ml_model.classes_
            prob_dict = {cls: prob for cls, prob in zip(classes, proba)}
        else:
            confidence = 0.8  # Default per modelli senza predict_proba
            prob_dict = {}
        
        # Mappa label inglese → italiano
        tipo_plagio = self.label_map.get(prediction, prediction)
        
        return {
            'tipo_plagio': tipo_plagio,
            'confidence': confidence,
            'probabilities': prob_dict,
            'prediction_raw': prediction
        }
        
    def kb_reasoning(self, ml_output, context):
        """
        Reasoning con Knowledge Base
        
        Args:
            ml_output: risultato predizione ML
            context: dizionario con parametri contestuali
        
        Returns:
            dict: risultato reasoning KB
        """
        # Estrai parametri
        tipo_ml = ml_output['tipo_plagio']
        confidence = ml_output['confidence']
        
        # Calcola similarità da confidence
        similarita = confidence
        
        # Estrai contesto
        percentuale = context.get('percentuale', 20)
        disciplina = context.get('disciplina', 'informatica')
        livello = context.get('livello_studio', 'bachelor')
        citazione = context.get('citazione', 'no')
        match_esatto = context.get('match_esatto', 'no')
        fonti_multiple = context.get('fonti_multiple', 'no')
        qualita_parafrasi = context.get('qualita_parafrasi', 'media')
        fonte_dichiarata = context.get('fonte_dichiarata', 'no')
        idea_originale = context.get('idea_originale', 'yes')
        sim_concettuale = context.get('sim_concettuale', 0.4)
        
        # Query Prolog
        query = f"""
        analizza_testo({similarita}, {citazione}, {match_esatto}, {fonti_multiple},
                      {qualita_parafrasi}, {sim_concettuale}, {fonte_dichiarata},
                      {idea_originale}, {percentuale}, {disciplina}, {livello},
                      TipoPlagio, Gravita, Raccomandazione, RaccDisciplina)
        """
        
        try:
            results = list(self.prolog.query(query))
            
            if results:
                result = results[0]
                return {
                    'tipo_kb': result['TipoPlagio'],
                    'gravita': result['Gravita'],
                    'raccomandazione': result['Raccomandazione'],
                    'racc_disciplina': result['RaccDisciplina'],
                    'kb_success': True
                }
            else:
                return {'kb_success': False, 'error': 'No results from KB'}
                
        except Exception as e:
            return {'kb_success': False, 'error': str(e)}
            
    def hybrid_decision(self, ml_output, kb_output):
        """
        Combina output ML e KB per decisione finale
        
        Decision logic:
        - Se ML e KB concordano → alta confidence
        - Se discordano → KB vince (più conservativo)
        - KB fornisce sempre gravità e raccomandazioni
        """
        if not kb_output.get('kb_success'):
            # Solo ML disponibile
            return {
                'tipo_finale': ml_output['tipo_plagio'],
                'confidence': ml_output['confidence'],
                'source': 'ml_only',
                'gravita': 'unknown',
                'raccomandazione': 'Verificare manualmente',
                'agreement': False
            }
        
        tipo_ml = ml_output['tipo_plagio']
        tipo_kb = kb_output['tipo_kb']
        
        # Verifica accordo
        agreement = (tipo_ml == tipo_kb)
        
        # Decision logic
        if agreement:
            # Concordano: alta confidence
            tipo_finale = tipo_ml
            confidence_finale = ml_output['confidence'] * 1.1  # Boost
            confidence_finale = min(confidence_finale, 1.0)
            source = 'ml+kb_agreement'
        else:
            # Discordano: KB vince (più conservativo)
            tipo_finale = tipo_kb
            confidence_finale = ml_output['confidence'] * 0.9  # Penalità
            source = 'kb_override'
        
        return {
            'tipo_finale': tipo_finale,
            'tipo_ml': tipo_ml,
            'tipo_kb': tipo_kb,
            'confidence': confidence_finale,
            'source': source,
            'agreement': agreement,
            'gravita': kb_output['gravita'],
            'raccomandazione': kb_output['raccomandazione'],
            'racc_disciplina': kb_output.get('racc_disciplina', '')
        }
        
    def analyze(self, text, context=None):
        """
        Analisi completa con sistema ibrido
        
        Args:
            text: testo da analizzare
            context: parametri contestuali (opzionali)
        
        Returns:
            dict: risultato analisi completa
        """
        if context is None:
            context = {}
        
        print("\n" + "="*80)
        print("🔍 ANALISI IBRIDA IN CORSO...")
        print("="*80)
        
        # Step 1: Predizione ML
        print("\n🤖 Step 1: Predizione Machine Learning...")
        ml_output = self.ml_predict(text)
        print(f"   Tipo: {ml_output['tipo_plagio']}")
        print(f"   Confidence: {ml_output['confidence']:.2%}")
        
        # Step 2: Reasoning KB
        print("\n🧠 Step 2: Reasoning Knowledge Base...")
        kb_output = self.kb_reasoning(ml_output, context)
        
        if kb_output.get('kb_success'):
            print(f"   Tipo: {kb_output['tipo_kb']}")
            print(f"   Gravità: {kb_output['gravita']}")
        else:
            print(f"   ⚠️ KB Error: {kb_output.get('error')}")
        
        # Step 3: Decisione ibrida
        print("\n⚖️ Step 3: Decisione Ibrida...")
        final_output = self.hybrid_decision(ml_output, kb_output)
        print(f"   Tipo Finale: {final_output['tipo_finale']}")
        print(f"   Source: {final_output['source']}")
        print(f"   Agreement: {final_output['agreement']}")
        
        # Combina tutti i risultati
        complete_result = {
            'text': text,
            'ml_output': ml_output,
            'kb_output': kb_output,
            'final_decision': final_output
        }
        
        print("\n✅ Analisi completata!")
        print("="*80)
        
        return complete_result
        
    def explain_decision(self, result):
        """Spiega la decisione presa dal sistema"""
        final = result['final_decision']
        ml = result['ml_output']
        kb = result['kb_output']
        
        explanation = []
        explanation.append("\n" + "="*80)
        explanation.append("💡 SPIEGAZIONE DELLA DECISIONE")
        explanation.append("="*80 + "\n")
        
        # ML prediction
        explanation.append("🤖 MACHINE LEARNING:")
        explanation.append(f"   Predizione: {ml['tipo_plagio']}")
        explanation.append(f"   Confidence: {ml['confidence']:.2%}")
        
        # KB reasoning
        if kb.get('kb_success'):
            explanation.append("\n🧠 KNOWLEDGE BASE:")
            explanation.append(f"   Tipo: {kb['tipo_kb']}")
            explanation.append(f"   Gravità: {kb['gravita']}")
            
            # Agreement
            explanation.append("\n⚖️ ACCORDO:")
            if final['agreement']:
                explanation.append("   ✅ ML e KB concordano")
                explanation.append("   → Confidence aumentata del 10%")
            else:
                explanation.append("   ⚠️ ML e KB discordano")
                explanation.append(f"   → KB prevale (più conservativo)")
                explanation.append(f"   → ML diceva: {ml['tipo_plagio']}")
                explanation.append(f"   → KB dice: {kb['tipo_kb']}")
        
        # Final decision
        explanation.append("\n🎯 DECISIONE FINALE:")
        explanation.append(f"   Tipo: {final['tipo_finale'].upper()}")
        explanation.append(f"   Gravità: {final['gravita'].upper()}")
        explanation.append(f"   Confidence: {final['confidence']:.2%}")
        
        # Recommendations
        explanation.append("\n💡 RACCOMANDAZIONI:")
        explanation.append(f"   {final['raccomandazione']}")
        if final.get('racc_disciplina'):
            explanation.append(f"\n   Specifica: {final['racc_disciplina']}")
        
        explanation.append("\n" + "="*80)
        
        return "\n".join(explanation)

def demo():
    """Demo del sistema ibrido"""
    print("\n🚀 DEMO SISTEMA IBRIDO ML+KB")
    print("="*80)
    
    # Inizializza sistema
    system = HybridPlagiarismSystem()
    
    # Testi di esempio
    test_cases = [
        {
            'text': 'Artificial intelligence is transforming modern society through machine learning algorithms that can process vast amounts of data.',
            'context': {
                'percentuale': 85,
                'disciplina': 'informatica',
                'livello_studio': 'bachelor',
                'citazione': 'no',
                'match_esatto': 'yes'
            }
        },
        {
            'text': 'Research has shown that innovation can lead to improved outcomes in various domains.',
            'context': {
                'percentuale': 15,
                'disciplina': 'medicina',
                'livello_studio': 'phd',
                'citazione': 'corretta'
            }
        }
    ]
    
    # Analizza casi
    for i, case in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"TEST CASE {i}")
        print(f"{'='*80}")
        print(f"Testo: {case['text'][:100]}...")
        
        result = system.analyze(case['text'], case['context'])
        explanation = system.explain_decision(result)
        print(explanation)

if __name__ == "__main__":
    demo()