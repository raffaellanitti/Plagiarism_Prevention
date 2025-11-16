"""
Training e Validazione Robusta con Multiple Runs e K-Fold Cross-Validation
Questo script implementa una valutazione rigorosa dei modelli di ML:
- K-fold cross-validation (10-fold)
- Multiple runs (5 ripetizioni)
- Calcolo medie e deviazioni standard
- Comparazione statistica tra modelli

Autore: [Raffaella Nitti]
Matricola: [796132]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import joblib
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🤖 PLAGIARISM DETECTION - TRAINING E VALIDAZIONE ROBUSTA")
print("="*80)

# ==================== CARICAMENTO DATI ====================
print("\n📂 Caricamento dati preprocessati...")

try:
    X_train = np.load('../dataset/X_train.npy', allow_pickle=True)
    X_test = np.load('../dataset/X_test.npy', allow_pickle=True)
    y_train = np.load('../dataset/y_train.npy', allow_pickle=True)
    y_test = np.load('../dataset/y_test.npy', allow_pickle=True)
    print("✅ Dati caricati con successo!")
    print(f"📊 Training set: {X_train.shape[0]} campioni, {X_train.shape[1]} features")
    print(f"📊 Test set: {X_test.shape[0]} campioni")
except FileNotFoundError:
    print("❌ ERRORE: Dati preprocessati non trovati!")
    print("Esegui prima 'preprocessing.py' per generare i file .npy")
    exit(1)

# ==================== DEFINIZIONE MODELLI ====================
print("\n" + "="*80)
print("🔧 DEFINIZIONE MODELLI")
print("="*80)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Naive Bayes': MultinomialNB()
}

print(f"\n📋 Modelli da testare: {len(models)}")
for name in models.keys():
    print(f"   - {name}")

# ==================== K-FOLD CROSS-VALIDATION ====================
print("\n" + "="*80)
print("🔄 K-FOLD CROSS-VALIDATION (10-fold, 5 ripetizioni)")
print("="*80)

# Parametri cross-validation
n_splits = 10
n_repeats = 5
scoring_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

# Dizionario per salvare risultati
cv_results = {model_name: {metric: [] for metric in scoring_metrics} 
              for model_name in models.keys()}

print(f"\n⚙️ Configurazione:")
print(f"   - K-folds: {n_splits}")
print(f"   - Ripetizioni: {n_repeats}")
print(f"   - Metriche: {', '.join(scoring_metrics)}")

# Esegui cross-validation per ogni modello
for model_name, model in tqdm(models.items(), desc="Valutazione modelli"):
    print(f"\n🔍 Valutazione: {model_name}")
    
    for repeat in range(n_repeats):
        # StratifiedKFold per mantenere distribuzione classi
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=repeat)
        
        for metric in scoring_metrics:
            scores = cross_val_score(model, X_train, y_train, 
                                   cv=skf, scoring=metric, n_jobs=-1)
            cv_results[model_name][metric].append(scores.mean())

# Calcola statistiche (media e std)
cv_stats = {}
for model_name in models.keys():
    cv_stats[model_name] = {}
    for metric in scoring_metrics:
        scores = cv_results[model_name][metric]
        cv_stats[model_name][metric] = {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores)
        }

# ==================== VISUALIZZAZIONE RISULTATI CV ====================
print("\n" + "="*80)
print("📊 RISULTATI CROSS-VALIDATION")
print("="*80)

# Tabella risultati
results_df = []
for model_name in models.keys():
    row = {'Model': model_name}
    for metric in scoring_metrics:
        mean = cv_stats[model_name][metric]['mean']
        std = cv_stats[model_name][metric]['std']
        row[f'{metric.replace("_macro", "").title()}'] = f'{mean:.4f} ± {std:.4f}'
    results_df.append(row)

results_df = pd.DataFrame(results_df)
print("\n" + results_df.to_string(index=False))

# Salva tabella
results_df.to_csv('../dataset/cv_results.csv', index=False)
print("\n✅ Risultati salvati in 'dataset/cv_results.csv'")

# Grafico comparativo
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Comparazione Modelli - Cross-Validation Results', 
             fontsize=16, fontweight='bold')

for idx, metric in enumerate(scoring_metrics):
    ax = axes[idx // 2, idx % 2]
    
    means = [cv_stats[m][metric]['mean'] for m in models.keys()]
    stds = [cv_stats[m][metric]['std'] for m in models.keys()]
    
    x_pos = np.arange(len(models))
    ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color='skyblue', 
           edgecolor='black', linewidth=1.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models.keys(), rotation=45, ha='right')
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title(f'{metric.replace("_macro", "").title()}', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig('../img/cv_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Grafico comparazione salvato in 'img/cv_comparison.png'")
plt.show()

# ==================== TRAINING FINALE SU TRAINING SET ====================
print("\n" + "="*80)
print("🎓 TRAINING FINALE SU INTERO TRAINING SET")
print("="*80)

trained_models = {}
for model_name, model in tqdm(models.items(), desc="Training modelli"):
    model.fit(X_train, y_train)
    trained_models[model_name] = model
    
    # Salva modello
    joblib.dump(model, f'../dataset/model_{model_name.replace(" ", "_").lower()}.pkl')

print("\n✅ Tutti i modelli addestrati e salvati!")

# ==================== VALUTAZIONE SU TEST SET ====================
print("\n" + "="*80)
print("🧪 VALUTAZIONE SU TEST SET")
print("="*80)

test_results = []
for model_name, model in trained_models.items():
    print(f"\n📊 Valutazione: {model_name}")
    
    # Predizioni
    y_pred = model.predict(X_test)
    
    # Metriche
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    
    test_results.append({
        'Model': model_name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1
    })
    
    print(f"   Accuracy:  {acc:.4f}")
    print(f"   Precision: {prec:.4f}")
    print(f"   Recall:    {rec:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=np.unique(y_test), 
                yticklabels=np.unique(y_test))
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'../img/confusion_matrix_{model_name.replace(" ", "_").lower()}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

print("\n✅ Confusion matrices salvate in 'img/'")

# Tabella risultati test
test_results_df = pd.DataFrame(test_results)
print("\n" + "="*80)
print("📋 RISULTATI TEST SET")
print("="*80)
print("\n" + test_results_df.to_string(index=False))

test_results_df.to_csv('../dataset/test_results.csv', index=False)
print("\n✅ Risultati test salvati in 'dataset/test_results.csv'")

# ==================== BEST MODEL ====================
best_model_name = test_results_df.loc[test_results_df['F1-Score'].idxmax(), 'Model']
best_f1 = test_results_df['F1-Score'].max()

print("\n" + "="*80)
print("🏆 MIGLIOR MODELLO")
print("="*80)
print(f"\n🥇 {best_model_name}")
print(f"   F1-Score: {best_f1:.4f}")

# Salva il miglior modello separatamente
best_model = trained_models[best_model_name]
joblib.dump(best_model, '../dataset/best_model.pkl')
print(f"\n✅ Miglior modello salvato in 'dataset/best_model.pkl'")

# ==================== CLASSIFICATION REPORT ====================
print("\n" + "="*80)
print(f"📊 CLASSIFICATION REPORT - {best_model_name}")
print("="*80)

y_pred_best = best_model.predict(X_test)
print("\n" + classification_report(y_test, y_pred_best))

# ==================== RIEPILOGO FINALE ====================
print("\n" + "="*80)
print("✅ TRAINING E VALUTAZIONE COMPLETATI!")
print("="*80)

print("\n📁 File generati:")
print("   - dataset/cv_results.csv (risultati cross-validation)")
print("   - dataset/test_results.csv (risultati test set)")
print("   - dataset/model_*.pkl (tutti i modelli addestrati)")
print("   - dataset/best_model.pkl (miglior modello)")
print("   - img/cv_comparison.png (comparazione cross-validation)")
print("   - img/confusion_matrix_*.png (matrici di confusione)")

print("\n🎯 Prossimo step: ottimizzazione iperparametri con 'optimized_*.py'")
print("="*80)