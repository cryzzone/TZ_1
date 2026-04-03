"""
Модель кластеризации для анализа структуры бизнеса
===================================================
Алгоритмы: K-Means, Иерархическая кластеризация
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings

warnings.filterwarnings("ignore")

# Загрузка и подготовка данных
print("=" * 60)
print("1. ПОДГОТОВКА ДАННЫХ")
print("=" * 60)

df = pd.read_csv("PP13_ISP23V_clustering.csv")
print(f"\nИсходные данные: {df.shape[0]} строк, {df.shape[1]} признаков")
print(f"Пропуски:\n{df.isnull().sum()}")

# Очистка: удаление пропусков (если есть)
df_clean = df.dropna()
print(f"После очистки: {df_clean.shape[0]} строк")

# Анализ корреляций
corr_matrix = df_clean.corr()
print(f"\nКорреляционная матрица:")
print(corr_matrix.round(3))

# Визуализация корреляций
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".3f", vmin=-1, vmax=1)
plt.title("Корреляционная матрица признаков")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=150)
print("Сохранено: correlation_heatmap.png")

# Масштабирование признаков
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean)
feature_names = df_clean.columns.tolist()
print(f"\nМасштабирование выполнено (StandardScaler)")
print(f"Признаки: {feature_names}")

# Определение оптимального числа кластеров
print("\n" + "=" * 60)
print("2. ОПРЕДЕЛЕНИЕ ЧИСЛА КЛАСТЕРОВ")
print("=" * 60)

k_range = range(2, 11)
inertias = []
silhouette_scores = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

# Метод локтя
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(k_range, inertias, "bo-", linewidth=2, markersize=8)
plt.xlabel("Количество кластеров (k)", fontsize=12)
plt.ylabel("WCSS (инерция)", fontsize=12)
plt.title("Метод локтя (Elbow Method)", fontsize=13, fontweight="bold")
plt.grid(True, alpha=0.3)

# Силуэтный коэффициент
plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette_scores, "ro-", linewidth=2, markersize=8)
plt.xlabel("Количество кластеров (k)", fontsize=12)
plt.ylabel("Silhouette Score", fontsize=12)
plt.title("Силуэтный коэффициент", fontsize=13, fontweight="bold")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("elbow_and_silhouette.png", dpi=150)
print("Сохранено: elbow_and_silhouette.png")

# Автоматический выбор лучшего k по силуэтному коэффициенту
best_k = k_range[np.argmax(silhouette_scores)]
print(f"\nЛучший silhouette score: {max(silhouette_scores):.4f} при k = {best_k}")
print(f"Выбранное число кластеров: {best_k}")

# K-Means кластеризация
print("\n" + "=" * 60)
print("3. K-MEANS КЛАСТЕРИЗАЦИЯ")
print("=" * 60)

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df_clean["cluster_km"] = kmeans.fit_predict(X_scaled)

print(f"\nРазмер кластеров (K-Means):")
print(df_clean["cluster_km"].value_counts().sort_index())

# Характеристики кластеров
print(f"\nХарактеристики кластеров (K-Means):")
cluster_stats_km = df_clean.groupby("cluster_km").mean()
print(cluster_stats_km.round(2))

# Иерархическая кластеризация
print("\n" + "=" * 60)
print("4. ИЕРАРХИЧЕСКАЯ КЛАСТЕРИЗАЦИЯ")
print("=" * 60)

hierarchical = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
df_clean["cluster_hc"] = hierarchical.fit_predict(X_scaled)

print(f"\nРазмер кластеров (Иерархическая):")
print(df_clean["cluster_hc"].value_counts().sort_index())

# Характеристики кластеров
print(f"\nХарактеристики кластеров (Иерархическая):")
cluster_stats_hc = df_clean.groupby("cluster_hc").mean()
print(cluster_stats_hc.round(2))

# Дендрограмма
linkage_matrix = linkage(X_scaled, method="ward")
plt.figure(figsize=(12, 6))
dendro = dendrogram(linkage_matrix, leaf_font_size=8, color_threshold=150)
plt.title("Дендрограмма (Иерархическая кластеризация)", fontsize=13, fontweight="bold")
plt.xlabel("Объекты", fontsize=12)
plt.ylabel("Расстояние", fontsize=12)
plt.axhline(y=150, color="r", linestyle="--", alpha=0.5, label=f"Порог разделения (k={best_k})")
plt.legend()
plt.tight_layout()
plt.savefig("dendrogram.png", dpi=150)
print("\nСохранено: dendrogram.png")

# Визуализация кластеров (2D через PCA)
print("\n" + "=" * 60)
print("5. ВИЗУАЛИЗАЦИЯ КЛАСТЕРОВ (PCA 2D)")
print("=" * 60)

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(f"Объяснённая дисперсия PCA: {pca.explained_variance_ratio_.round(4)}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# K-Means
scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=df_clean["cluster_km"],
                           cmap="viridis", alpha=0.7, s=50, edgecolors="k")
axes[0].set_title(f"K-Means (k={best_k})", fontsize=13, fontweight="bold")
axes[0].set_xlabel("PC1")
axes[0].set_ylabel("PC2")
plt.colorbar(scatter1, ax=axes[0], label="Кластер")

# Иерархическая
scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=df_clean["cluster_hc"],
                           cmap="plasma", alpha=0.7, s=50, edgecolors="k")
axes[1].set_title(f"Иерархическая (k={best_k})", fontsize=13, fontweight="bold")
axes[1].set_xlabel("PC1")
axes[1].set_ylabel("PC2")
plt.colorbar(scatter2, ax=axes[1], label="Кластер")

plt.tight_layout()
plt.savefig("clusters_2d.png", dpi=150)
print("Сохранено: clusters_2d.png")

# Сравнение алгоритмов
print("\n" + "=" * 60)
print("6. СРАВНЕНИЕ АЛГОРИТМОВ")
print("=" * 60)

sil_km = silhouette_score(X_scaled, df_clean["cluster_km"])
sil_hc = silhouette_score(X_scaled, df_clean["cluster_hc"])

print(f"\nK-Means:           silhouette = {sil_km:.4f}")
print(f"Иерархическая:    silhouette = {sil_hc:.4f}")

if sil_km >= sil_hc:
    best_algo = "K-Means"
    df_clean["cluster"] = df_clean["cluster_km"]
else:
    best_algo = "Иерархическая"
    df_clean["cluster"] = df_clean["cluster_hc"]

print(f"\nЛучший алгоритм: {best_algo}")

# Интерпретация кластеров
print("\n" + "=" * 60)
print("7. ИНТЕРПРЕТАЦИЯ КЛАСТЕРОВ")
print("=" * 60)

final_stats = df_clean.groupby("cluster").mean()
final_counts = df_clean["cluster"].value_counts().sort_index()

for cl in sorted(df_clean["cluster"].unique()):
    mask = df_clean["cluster"] == cl
    n = mask.sum()
    pct = n / len(df_clean) * 100
    print(f"\n{'─' * 50}")
    print(f"КЛАСТЕР {cl}: {n} клиентов ({pct:.1f}%)")
    print(f"  Возраст:            {df_clean.loc[mask, 'customer_age'].mean():.1f} лет")
    print(f"  Годовой доход:      {df_clean.loc[mask, 'annual_income'].mean():.0f} руб.")
    print(f"  Оценка расходов:    {df_clean.loc[mask, 'spending_score'].mean():.1f}")
    print(f"  Частота покупок:    {df_clean.loc[mask, 'purchase_frequency_per_month'].mean():.1f} раз/мес")

# Сохранение результатов
df_clean.to_csv("clustering_results.csv", index=False)
print(f"\n{'=' * 60}")
print("РЕЗУЛЬТАТЫ СОХРАНЕНЫ: clustering_results.csv")
print(f"{'=' * 60}")
