"""
Ottimizzazione Iperparametri SVM con GridSearchCV
Questo script ottimizza il modello SVM per migliorare le prestazioni

Autore: [Nome Cognome]
Matricola: [XXXXXX]
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🎯 OTTIMIZZAZIONE SVM - GridSearchCV")
print("="*80)

# ==================== CARICAMENTO DATI ====================
print("\n📂 Caricamento dati...")

X_train = np.load('../dataset/X_train.npy', allow_pickle=True)
X_test = np.load('../dataset/X_test.npy', allow_pickle=True)
y_train = np.load('../dataset/y_train.npy', allow_pickle=True)
y_test = np.load('../dataset/y_test.npy', allow_pickle=True)

print(f"✅ Training set: {X_train.shape[0]} campioni, {X_train.shape[1]} features")
print(f"✅ Test set: {X_test.shape[0]} campioni")

# ==================== MODELLO BASELINE ====================
print("\n" + "="*80)
print("📊 VALUTAZIONE MODELLO BASELINE")
print("="*80)

baseline_model = SVC(kernel='rbf', random_state=42)
baseline_model.fit(X_train, y_train)

baseline_score = baseline_model.score(X_test, y_test)
print(f"\n🎯 Accuracy Baseline: {baseline_score:.4f}")

# Cross-validation baseline
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
baseline_cv_scores = cross_val_score(baseline_model, X_train, y_train, cv=cv, scoring='f1_macro')
print(f"📊 F1-Score CV Baseline: {baseline_cv_scores.mean():.4f} ± {baseline_cv_scores.std():.4f}")

# ==================== GRID SEARCH ====================
print("\n" + "="*80)
print("🔍 GRID SEARCH - Ottimizzazione Iperparametri")
print("="*80)

# Definizione spazio di ricerca
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'kernel': ['rbf', 'poly', 'sigmoid'],
    'degree': [2, 3, 4]  # Solo per kernel poly
}

print("\n📋 Spazio di ricerca:")
for param, values in param_grid.items():
    print(f"   - {param}: {values}")

# Nota: degree è rilevante solo per poly, ma GridSearch lo ignora per altri kernel
total_combinations = len(param_grid['C']) * len(param_grid['gamma']) * len(param_grid['kernel']) * len(param_grid['degree'])
print(f"\n🔢 Combinazioni da testare (approssimate): ~{total_combinations//3}")  # Diviso 3 perché degree è solo per poly

# Grid Search
print("\n⚙️ Esecuzione GridSearchCV (5-fold CV)...")
print("⏳ Questo potrebbe richiedere alcuni minuti...\n")

start_time = time.time()

grid_search = GridSearchCV(
    estimator=SVC(random_state=42),
    param_grid=param_grid,
    cv=cv,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)

elapsed_time = time.time() - start_time
print(f"\n✅ Grid Search completato in {elapsed_time/60:.2f} minuti")

# ==================== RISULTATI OTTIMIZZAZIONE ====================
print("\n" + "="*80)
print("🏆 RISULTATI OTTIMIZZAZIONE")
print("="*80)

print("\n🥇 Migliori iperparametri trovati:")
for param, value in grid_search.best_params_.items():
    print(f"   - {param}: {value}")

print(f"\n📊 Best CV F1-Score: {grid_search.best_score_:.4f}")

# ==================== VALUTAZIONE MODELLO OTTIMIZZATO ====================
print("\n" + "="*80)
print("🧪 VALUTAZIONE MODELLO OTTIMIZZATO SU TEST SET")
print("="*80)

best_model = grid_search.best_estimator_
test_score = best_model.score(X_test, y_test)

print(f"\n🎯 Accuracy Test: {test_score:.4f}")

y_pred = best_model.predict(X_test)

# Classification Report
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y_test),
            yticklabels=np.unique(y_test))
plt.title('Confusion Matrix - SVM Ottimizzato', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('../img/confusion_matrix_svm_optimized.png', dpi=300, bbox_inches='tight')
print("\n✅ Confusion matrix salvata in 'img/confusion_matrix_svm_optimized.png'")
plt.close()

# ==================== COMPARAZIONE BASELINE vs OTTIMIZZATO ====================
print("\n" + "="*80)
print("📈 COMPARAZIONE: BASELINE vs OTTIMIZZATO")
print("="*80)

comparison = pd.DataFrame({
    'Model': ['Baseline', 'Ottimizzato'],
    'CV F1-Score': [baseline_cv_scores.mean(), grid_search.best_score_],
    'Test Accuracy': [baseline_score, test_score],
    'Improvement': [0, test_score - baseline_score]
})

print("\n" + comparison.to_string(index=False))

# Grafico comparazione
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# CV Scores
ax[0].bar(['Baseline', 'Ottimizzato'], 
          [baseline_cv_scores.mean(), grid_search.best_score_],
          color=['skyblue', 'orange'], edgecolor='black', linewidth=2)
ax[0].set_ylabel('F1-Score', fontsize=12)
ax[0].set_title('Cross-Validation F1-Score', fontsize=14, fontweight='bold')
ax[0].set_ylim([0.9, 1.0])
ax[0].grid(axis='y', alpha=0.3)

# Test Accuracy
ax[1].bar(['Baseline', 'Ottimizzato'], 
          [baseline_score, test_score],
          color=['skyblue', 'orange'], edgecolor='black', linewidth=2)
ax[1].set_ylabel('Accuracy', fontsize=12)
ax[1].set_title('Test Set Accuracy', fontsize=14, fontweight='bold')
ax[1].set_ylim([0.9, 1.0])
ax[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('../img/svm_baseline_vs_optimized.png', dpi=300, bbox_inches='tight')
print("\n✅ Grafico comparazione salvato in 'img/svm_baseline_vs_optimized.png'")
plt.close()

# ==================== ANALISI PARAMETRI ====================
print("\n" + "="*80)
print("📊 ANALISI IMPATTO PARAMETRI")
print("="*80)

# Estrai risultati del grid search
cv_results = pd.DataFrame(grid_search.cv_results_)

# Analizza impatto di C
print("\n🔍 Impatto parametro C (regolarizzazione):")
c_impact = cv_results.groupby('param_C')['mean_test_score'].mean().sort_values(ascending=False)
print(c_impact.to_string())

# Analizza impatto di gamma
print("\n🔍 Impatto parametro gamma:")
gamma_impact = cv_results.groupby('param_gamma')['mean_test_score'].mean().sort_values(ascending=False)
print(gamma_impact.to_string())

# Analizza impatto del kernel
print("\n🔍 Impatto tipo kernel:")
kernel_impact = cv_results.groupby('param_kernel')['mean_test_score'].mean().sort_values(ascending=False)
print(kernel_impact.to_string())

# ==================== SALVATAGGIO MODELLO ====================
print("\n" + "="*80)
print("💾 SALVATAGGIO MODELLO OTTIMIZZATO")
print("="*80)

joblib.dump(best_model, '../dataset/svm_optimized.pkl')
print("\n✅ Modello ottimizzato salvato in 'dataset/svm_optimized.pkl'")

# Salva anche i risultati
results = {
    'best_params': grid_search.best_params_,
    'best_cv_score': grid_search.best_score_,
    'test_accuracy': test_score,
    'baseline_cv_score': baseline_cv_scores.mean(),
    'baseline_test_accuracy': baseline_score,
    'improvement': test_score - baseline_score
}

import json
with open('../dataset/svm_optimization_results.json', 'w') as f:
    json.dump(results, f, indent=4)

print("✅ Risultati salvati in 'dataset/svm_optimization_results.json'")

# ==================== RIEPILOGO ====================
print("\n" + "="*80)
print("✅ OTTIMIZZAZIONE SVM COMPLETATA!")
print("="*80)

print(f"\n📊 Miglioramento ottenuto: +{(test_score - baseline_score)*100:.2f}%")
print(f"🎯 Accuracy finale: {test_score:.4f}")

print("\n📁 File generati:")
print("   - dataset/svm_optimized.pkl")
print("   - dataset/svm_optimization_results.json")
print("   - img/confusion_matrix_svm_optimized.png")
print("   - img/svm_baseline_vs_optimized.png")

print("\n🎯 Prossimo step: eseguire 'optimized_knn.py'")
print("="*80)