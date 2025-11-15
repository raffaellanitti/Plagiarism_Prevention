"""
Test del Sistema Ibrido con Casi Complessi
Dimostra come ML+KB risolvono casi difficili

Autore: [Raffaella Nitti]
Matricola: [796132]
"""

import sys
sys.path.append('..')
from integration.hybrid_system import HybridPlagiarismSystem
import pandas as pd

def test_complex_cases():
    """Test con casi complessi che dimostrano il valore dell'integrazione"""
    
    print("="*80)
    print("🧪 TEST CASI COMPLESSI - SISTEMA IBRIDO")
    print("="*80)
    
    # Inizializza sistema
    system = HybridPlagiarismSystem()
    
    # Casi di test complessi
    test_cases = [
        {
            'name': 'CASO 1: Formula Standard in Ingegneria',
            'description': 'Formula matematica comune - ML potrebbe classificare come plagio, KB sa che è OK',
            'text': 'The formula F = ma represents Newton second law of motion',
            'context': {
                'percentuale': 90,  # Alta similarità
                'disciplina': 'ingegneria',
                'livello_studio': 'bachelor',
                'citazione': 'no',
                'match_esatto': 'yes',
                'fonti_multiple': 'no',
                'qualita_parafrasi': 'scarsa',
                'fonte_dichiarata': 'no',
                'idea_originale': 'no',
                'sim_concettuale': 0.9
            }
        },
        {
            'name': 'CASO 2: Plagio Mosaico Sofisticato',
            'description': 'Pezzi da fonti diverse - KB rileva pattern che ML potrebbe perdere',
            'text': 'Climate change represents significant challenge humanity faces DNA molecules contain genetic instructions quantum computing promises revolutionize cryptography',
            'context': {
                'percentuale': 60,
                'disciplina': 'filosofia',
                'livello_studio': 'phd',
                'citazione': 'no',
                'match_esatto': 'no',
                'fonti_multiple': 'yes',
                'qualita_parafrasi': 'media',
                'fonte_dichiarata': 'no',
                'idea_originale': 'no',
                'sim_concettuale': 0.7
            }
        },
        {
            'name': 'CASO 3: Parafrasi con Citazione',
            'description': 'Similarità alta ma citazione presente - KB valuta correttezza',
            'text': 'According to recent studies artificial intelligence transforms society through machine learning processing large data volumes',
            'context': {
                'percentuale': 75,
                'disciplina': 'informatica',
                'livello_studio': 'master',
                'citazione': 'yes',  # Citazione presente
                'match_esatto': 'no',
                'fonti_multiple': 'no',
                'qualita_parafrasi': 'scarsa',
                'fonte_dichiarata': 'yes',
                'idea_originale': 'no',
                'sim_concettuale': 0.8
            }
        },
        {
            'name': 'CASO 4: Testo Originale con Terminologia Comune',
            'description': 'ML potrebbe flag, KB riconosce terminologia standard',
            'text': 'Studies demonstrate innovation contributes significantly enhanced performance various modern applications technology',
            'context': {
                'percentuale': 25,
                'disciplina': 'medicina',
                'livello_studio': 'phd',
                'citazione': 'corretta',
                'match_esatto': 'no',
                'fonti_multiple': 'no',
                'qualita_parafrasi': 'buona',
                'fonte_dichiarata': 'yes',
                'idea_originale': 'yes',
                'sim_concettuale': 0.3
            }
        },
        {
            'name': 'CASO 5: Plagio Letterale Mascherato',
            'description': 'Alta similarità con piccole modifiche - entrambi devono rilevare',
            'text': 'Artificial intelligence transforms contemporary society via machine learning algorithms processing enormous data quantities',
            'context': {
                'percentuale': 95,
                'disciplina': 'letteratura',
                'livello_studio': 'bachelor',
                'citazione': 'no',
                'match_esatto': 'yes',
                'fonti_multiple': 'no',
                'qualita_parafrasi': 'scarsa',
                'fonte_dichiarata': 'no',
                'idea_originale': 'no',
                'sim_concettuale': 0.95
            }
        }
    ]
    
    # Risultati
    results = []
    
    for case in test_cases:
        print("\n" + "="*80)
        print(f"📋 {case['name']}")
        print("="*80)
        print(f"Descrizione: {case['description']}")
        print(f"Testo: {case['text'][:100]}...")
        print(f"Disciplina: {case['context']['disciplina']}")
        print(f"Livello: {case['context']['livello_studio']}")
        
        # Analizza
        result = system.analyze(case['text'], case['context'])
        
        # Estrai info chiave
        ml_tipo = result['ml_output']['tipo_plagio']
        ml_conf = result['ml_output']['confidence']
        
        if result['kb_output'].get('kb_success'):
            kb_tipo = result['kb_output']['tipo_kb']
            kb_grav = result['kb_output']['gravita']
        else:
            kb_tipo = 'ERROR'
            kb_grav = 'ERROR'
        
        final_tipo = result['final_decision']['tipo_finale']
        final_grav = result['final_decision']['gravita']
        agreement = result['final_decision']['agreement']
        source = result['final_decision']['source']
        
        # Stampa risultati
        print("\n📊 RISULTATI:")
        print(f"   ML predice: {ml_tipo} (conf: {ml_conf:.2%})")
        print(f"   KB dice: {kb_tipo} (gravità: {kb_grav})")
        print(f"   FINALE: {final_tipo} (gravità: {final_grav})")
        print(f"   Accordo ML-KB: {'✅ SÌ' if agreement else '⚠️ NO'}")
        print(f"   Decisione da: {source}")
        
        # Spiegazione
        explanation = system.explain_decision(result)
        print(explanation)
        
        # Salva per report
        results.append({
            'Caso': case['name'],
            'ML_Tipo': ml_tipo,
            'ML_Conf': f"{ml_conf:.2%}",
            'KB_Tipo': kb_tipo,
            'KB_Gravità': kb_grav,
            'Finale_Tipo': final_tipo,
            'Finale_Gravità': final_grav,
            'Accordo': 'Sì' if agreement else 'No',
            'Source': source
        })
    
    # Report finale
    print("\n" + "="*80)
    print("📊 REPORT FINALE - CONFRONTO CASI")
    print("="*80 + "\n")
    
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    # Statistiche
    print("\n" + "="*80)
    print("📈 STATISTICHE")
    print("="*80)
    
    total = len(results)
    agreements = sum(1 for r in results if r['Accordo'] == 'Sì')
    kb_overrides = sum(1 for r in results if r['Source'] == 'kb_override')
    
    print(f"\nTotale casi testati: {total}")
    print(f"Accordo ML-KB: {agreements}/{total} ({agreements/total*100:.1f}%)")
    print(f"KB override ML: {kb_overrides}/{total} ({kb_overrides/total*100:.1f}%)")
    
    print("\n💡 INSIGHT:")
    if kb_overrides > 0:
        print(f"   In {kb_overrides} casi, la KB ha corretto la predizione ML")
        print("   → Dimostra il valore del reasoning simbolico!")
    if agreements == total:
        print("   ML e KB concordano sempre")
        print("   → Sistema molto coerente!")
    else:
        print(f"   In {total - agreements} casi ML e KB discordano")
        print("   → Sistema ibrido gestisce l'incertezza!")
    
    print("\n" + "="*80)
    print("✅ TEST COMPLETATI!")
    print("="*80)
    
    # Salva report
    df.to_csv('../dataset/hybrid_system_test_results.csv', index=False)
    print("\n💾 Report salvato in 'dataset/hybrid_system_test_results.csv'")

if __name__ == "__main__":
    test_complex_cases()