# Gerekli kütüphaneleri içe aktar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score, davies_bouldin_score
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

# İris veri setini yükle
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Veriyi standartlaştır
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-means kümeleme uygula
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# Performans metriklerini hesapla
silhouette = silhouette_score(X_scaled, clusters)
ari = adjusted_rand_score(y, clusters)
nmi = normalized_mutual_info_score(y, clusters)
dbi = davies_bouldin_score(X_scaled, clusters)

print(f"Silhouette Score: {silhouette:.4f}")
print(f"Adjusted Rand Index: {ari:.4f}")
print(f"Normalized Mutual Information: {nmi:.4f}")
print(f"Davies-Bouldin Index: {dbi:.4f}")

# PCA ile veriyi 2 boyuta indirge ve görselleştir
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# K-means kümelerini görselleştir
plt.figure(figsize=(10, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', s=50, alpha=0.8)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', s=200, alpha=0.5, marker='X')
plt.title('K-means Kümeleme Sonuçları (PCA ile 2D)')
plt.xlabel('Birinci Temel Bileşen')
plt.ylabel('İkinci Temel Bileşen')
plt.colorbar(label='Küme')
plt.show()

# Gerçek etiketleri görselleştir
plt.figure(figsize=(10, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', s=50, alpha=0.8)
plt.title('Gerçek İris Sınıfları (PCA ile 2D)')
plt.xlabel('Birinci Temel Bileşen')
plt.ylabel('İkinci Temel Bileşen')
plt.colorbar(label='İris Sınıfı')
plt.show()

# Şimdi TensorFlow/Keras ile bir otoenkoder modeli oluşturalım
# Bu model verileri sıkıştırarak kümeleme için daha iyi özellikler çıkarabilir

# Otoenkoder modeli
input_dim = X_scaled.shape[1]
encoding_dim = 2  # 2 boyutlu gizli katman

# Enkoder
input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation='relu')(input_layer)

# Dekoder
decoder = Dense(input_dim, activation='sigmoid')(encoder)

# Tam otoenkoder
autoencoder = Model(inputs=input_layer, outputs=decoder)

# Sadece enkoder kısmı
encoder_model = Model(inputs=input_layer, outputs=encoder)

# Modeli derle
autoencoder.compile(optimizer=Adam(0.001), loss='mse')

# Otoenkoderi eğit
history = autoencoder.fit(
    X_scaled, X_scaled,
    epochs=100,
    batch_size=16,
    shuffle=True,
    validation_split=0.2,
    verbose=1
)

# Eğitim sürecini görselleştir
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.title('Otoenkoder Eğitim Süreci')
plt.xlabel('Epoch')
plt.ylabel('Kayıp (MSE)')
plt.legend()
plt.show()

# Özellik çıkarma için enkoderi kullan
encoded_features = encoder_model.predict(X_scaled)

# Otoenkoder özellikleri üzerinde K-means uygula
kmeans_ae = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters_ae = kmeans_ae.fit_predict(encoded_features)

# Otoenkoder özellikleri için performans metriklerini hesapla
silhouette_ae = silhouette_score(encoded_features, clusters_ae)
ari_ae = adjusted_rand_score(y, clusters_ae)
nmi_ae = normalized_mutual_info_score(y, clusters_ae)
dbi_ae = davies_bouldin_score(encoded_features, clusters_ae)

print("\n--- Otoenkoder Özellikleri ile Kümeleme Sonuçları ---")
print(f"Silhouette Score: {silhouette_ae:.4f}")
print(f"Adjusted Rand Index: {ari_ae:.4f}")
print(f"Normalized Mutual Information: {nmi_ae:.4f}")
print(f"Davies-Bouldin Index: {dbi_ae:.4f}")

# Otoenkoder özellikleri üzerinde küme sonuçlarını görselleştir
plt.figure(figsize=(10, 8))
plt.scatter(encoded_features[:, 0], encoded_features[:, 1], c=clusters_ae, cmap='viridis', s=50, alpha=0.8)
plt.scatter(kmeans_ae.cluster_centers_[:, 0], kmeans_ae.cluster_centers_[:, 1], c='red', s=200, alpha=0.5, marker='X')
plt.title('Otoenkoder + K-means Kümeleme Sonuçları')
plt.xlabel('Özellik 1')
plt.ylabel('Özellik 2')
plt.colorbar(label='Küme')
plt.show()

# Gerçek etiketlerle otoenkoder özelliklerini göster
plt.figure(figsize=(10, 8))
plt.scatter(encoded_features[:, 0], encoded_features[:, 1], c=y, cmap='viridis', s=50, alpha=0.8)
plt.title('Gerçek İris Sınıfları (Otoenkoder Özellikleri)')
plt.xlabel('Özellik 1')
plt.ylabel('Özellik 2')
plt.colorbar(label='İris Sınıfı')
plt.show()

# Her iki yaklaşımın karşılaştırma tablosunu oluştur
comparison = pd.DataFrame({
    'Metrik': ['Silhouette Score', 'Adjusted Rand Index', 'Normalized MI', 'Davies-Bouldin Index'],
    'K-means': [silhouette, ari, nmi, dbi],
    'Otoenkoder + K-means': [silhouette_ae, ari_ae, nmi_ae, dbi_ae]
})

print("\nKümeleme Performans Karşılaştırması:")
print(comparison)

# Çiçek özellikleri ve küme ilişkilerini görselleştir
df = pd.DataFrame(data=X, columns=feature_names)
df['Cluster'] = clusters_ae
df['Species'] = [target_names[i] for i in y]

plt.figure(figsize=(15, 10))
sns.pairplot(df, hue='Cluster', palette='viridis')
plt.suptitle('Özellikler ve Küme İlişkileri', y=1.02)
plt.show()