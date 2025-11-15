"""
Ottimizzazione Iperparametri KNN con GridSearchCV
Questo script ottimizza il modello K-Nearest Neighbors per migliorare le prestazioni

Autore: [Nome Cognome]
Matricola: [XXXXXX]
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🎯 OTTIMIZZAZIONE KNN - GridSearchCV")
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

baseline_model = KNeighborsClassifier(n_neighbors=5)
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
    'n_neighbors': [3, 5, 7, 9, 11, 15, 21],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski'],
    'p': [1, 2, 3],  # Parametro per Minkowski
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
}

print("\n📋 Spazio di ricerca:")
for param, values in param_grid.items():
    print(f"   - {param}: {values}")

total_combinations = np.prod([len(v) for v in param_grid.values()])
print(f"\n🔢 Combinazioni totali da testare: {total_combinations}")

# Grid Search
print("\n⚙️ Esecuzione GridSearchCV (5-fold CV)...")
print("⏳ Questo potrebbe richiedere alcuni minuti...\n")

start_time = time.time()

grid_search = GridSearchCV(
    estimator=KNeighborsClassifier(),
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
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=np.unique(y_test),
            yticklabels=np.unique(y_test))
plt.title('Confusion Matrix - KNN Ottimizzato', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('../img/confusion_matrix_knn_optimized.png', dpi=300, bbox_inches='tight')
print("\n✅ Confusion matrix salvata in 'img/confusion_matrix_knn_optimized.png'")
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
          color=['skyblue', 'mediumpurple'], edgecolor='black', linewidth=2)
ax[0].set_ylabel('F1-Score', fontsize=12)
ax[0].set_title('Cross-Validation F1-Score', fontsize=14, fontweight='bold')
ax[0].set_ylim([0.9, 1.0])
ax[0].grid(axis='y', alpha=0.3)

# Test Accuracy
ax[1].bar(['Baseline', 'Ottimizzato'], 
          [baseline_score, test_score],
          color=['skyblue', 'mediumpurple'], edgecolor='black', linewidth=2)
ax[1].set_ylabel('Accuracy', fontsize=12)
ax[1].set_title('Test Set Accuracy', fontsize=14, fontweight='bold')
ax[1].set_ylim([0.9, 1.0])
ax[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('../img/knn_baseline_vs_optimized.png', dpi=300, bbox_inches='tight')
print("\n✅ Grafico comparazione salvato in 'img/knn_baseline_vs_optimized.png'")
plt.close()

# ==================== ANALISI K OTTIMALE ====================
print("\n" + "="*80)
print("📊 ANALISI NUMERO DI NEIGHBORS (K)")
print("="*80)

# Test diversi valori di K per visualizzazione
k_values = range(1, 31)
k_scores = []

print("\n⚙️ Testando diversi valori di K...")
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, **{key: val for key, val in grid_search.best_params_.items() if key != 'n_neighbors'})
    scores = cross_val_score(knn, X_train, y_train, cv=cv, scoring='f1_macro')
    k_scores.append(scores.mean())

# Plot K vs Score
plt.figure(figsize=(12, 6))
plt.plot(k_values, k_scores, marker='o', linewidth=2, markersize=6)
plt.axvline(x=grid_search.best_params_['n_neighbors'], color='red', linestyle='--', 
            label=f"K ottimale = {grid_search.best_params_['n_neighbors']}")
plt.xlabel('Numero di Neighbors (K)', fontsize=12)
plt.ylabel('F1-Score (CV)', fontsize=12)
plt.title('Impatto del numero di Neighbors sulle prestazioni', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('../img/knn_k_analysis.png', dpi=300, bbox_inches='tight')
print("\n✅ Grafico analisi K salvato in 'img/knn_k_analysis.png'")
plt.close()

# ==================== ANALISI PARAMETRI ====================
print("\n" + "="*80)
print("📊 ANALISI IMPATTO PARAMETRI")
print("="*80)

# Estrai risultati del grid search
cv_results = pd.DataFrame(grid_search.cv_results_)

# Analizza impatto di weights
print("\n🔍 Impatto parametro weights:")
weights_impact = cv_results.groupby('param_weights')['mean_test_score'].mean().sort_values(ascending=False)
print(weights_impact.to_string())

# Analizza impatto di metric
print("\n🔍 Impatto tipo metrica:")
metric_impact = cv_results.groupby('param_metric')['mean_test_score'].mean().sort_values(ascending=False)
print(metric_impact.to_string())

# ==================== SALVATAGGIO MODELLO ====================
print("\n" + "="*80)
print("💾 SALVATAGGIO MODELLO OTTIMIZZATO")
print("="*80)

joblib.dump(best_model, '../dataset/knn_optimized.pkl')
print("\n✅ Modello ottimizzato salvato in 'dataset/knn_optimized.pkl'")

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
with open('../dataset/knn_optimization_results.json', 'w') as f:
    json.dump(results, f, indent=4)

print("✅ Risultati salvati in 'dataset/knn_optimization_results.json'")

# ==================== RIEPILOGO ====================
print("\n" + "="*80)
print("✅ OTTIMIZZAZIONE KNN COMPLETATA!")
print("="*80)

print(f"\n📊 Miglioramento ottenuto: +{(test_score - baseline_score)*100:.2f}%")
print(f"🎯 Accuracy finale: {test_score:.4f}")

print("\n📁 File generati:")
print("   - dataset/knn_optimized.pkl")
print("   - dataset/knn_optimization_results.json")
print("   - img/confusion_matrix_knn_optimized.png")
print("   - img/knn_baseline_vs_optimized.png")
print("   - img/knn_k_analysis.png")

print("\n🎉 FASE APPRENDIMENTO SUPERVISIONATO COMPLETATA!")
print("🎯 Prossima fase: Sistema Esperto in Prolog")
print("="*80)