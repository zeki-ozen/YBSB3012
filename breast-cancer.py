import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# Veriyi oku
df = pd.read_csv('breast-cancer.csv')

# Veri seti hakkında bilgi
print("Veri seti boyutu:", df.shape)
print("İlk 5 satır:")
print(df.head())

# id niteligini sil
df.drop(columns=['id'], inplace=True)

# Kategorik değişkenleri sayısala çevir
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = LabelEncoder().fit_transform(df[col])

# diagnosis hedef niteligi yap digerleri bagimsiz nitelikler
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Hedef değişkeni kategorik yap
num_classes = len(np.unique(y))
y_cat = to_categorical(y, num_classes)

# Veriyi eğitim ve test olarak ayır
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# Özellikleri ölçeklendir
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Modeli oluştur
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
   # Dropout(0.3),
    Dense(32, activation='relu'),
    #  Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Model özeti
model.summary()

# Modeli derle
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Erken durdurma için callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Modeli eğit
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.1,
    callbacks=[early_stopping]
)

# Test setinde değerlendirme
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Kaybı: {loss:.4f}')
print(f'Test Doğruluğu: {accuracy:.4f}')

# Eğitim sürecini görselleştir
plt.figure(figsize=(12, 5))

# Doğruluk grafiği
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.title('Model Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()

# Kayıp grafiği
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.title('Model Kaybı')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()

plt.tight_layout()
plt.show()