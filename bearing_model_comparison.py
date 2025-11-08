import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# ====================================================
# 1️⃣ Load Enhanced Dataset
# ====================================================
data = pd.read_csv(r"C:\Users\rinki.sharma\Downloads\enhanced_synthetic_bearing_data.csv")
print("✅ Enhanced Data Loaded Successfully")
print(data.head())

# ====================================================
# 2️⃣ Separate Features & Labels
# ====================================================
X = data.drop(columns=["label"])
y = data["label"].map({"Normal": 0, "Faulty": 1})

# Scale features for better model performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Separate Normal data (for training) and all data (for testing)
X_train = X_scaled[y == 0]  # only normal data
X_test = X_scaled           # all data for testing
y_test = y

print(f"\nTraining samples (Normal only): {X_train.shape[0]}")
print(f"Testing samples (Normal + Faulty): {X_test.shape[0]}")

# ====================================================
# 3️⃣ Autoencoder Model
# ====================================================
print("\n🔧 Training Autoencoder on Normal Data...")

input_dim = X_train.shape[1]
autoencoder = Sequential([
    Dense(16, activation='relu', input_shape=(input_dim,)),
    Dense(8, activation='relu'),
    Dense(16, activation='relu'),
    Dense(input_dim, activation='linear')
])

autoencoder.compile(optimizer='adam', loss='mse')

history = autoencoder.fit(
    X_train, X_train,
    epochs=60,
    batch_size=16,
    validation_split=0.1,
    verbose=0
)

# Reconstruction error
reconstructions = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)

# Set dynamic threshold (95th percentile of training error)
threshold = np.percentile(np.mean(np.power(X_train - autoencoder.predict(X_train), 2), axis=1), 95)
y_pred_ae = (mse > threshold).astype(int)

# ====================================================
# 4️⃣ Isolation Forest Model
# ====================================================
print("\n🌲 Training Isolation Forest on Normal Data...")

iso_forest = IsolationForest(contamination=0.1, random_state=42)
iso_forest.fit(X_train)

# Predict (-1 = anomaly, 1 = normal)
y_pred_if = iso_forest.predict(X_test)
y_pred_if = np.where(y_pred_if == -1, 1, 0)

# ====================================================
# 5️⃣ Evaluation Metrics
# ====================================================
def get_scores(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0)
    }

scores_ae = get_scores(y_test, y_pred_ae)
scores_if = get_scores(y_test, y_pred_if)

comparison = pd.DataFrame([scores_ae, scores_if], index=["Autoencoder", "Isolation Forest"])

print("\n📊 Autoencoder Performance:")
print(confusion_matrix(y_test, y_pred_ae))
print(classification_report(y_test, y_pred_ae, target_names=["Normal", "Faulty"]))

print("\n📊 Isolation Forest Performance:")
print(confusion_matrix(y_test, y_pred_if))
print(classification_report(y_test, y_pred_if, target_names=["Normal", "Faulty"]))

print("\n📈 Model Comparison:")
print(comparison)

best_model = comparison["F1 Score"].idxmax()
print(f"\n🏆 Best Model Based on F1 Score: {best_model}")

# ====================================================
# 6️⃣ Visualization
# ====================================================
plt.figure(figsize=(12, 5))
plt.hist(mse[y_test == 0], bins=30, alpha=0.6, label='Normal')
plt.hist(mse[y_test == 1], bins=30, alpha=0.6, label='Faulty')
plt.axvline(threshold, color='red', linestyle='--', label='Threshold')
plt.legend()
plt.title("Autoencoder Reconstruction Error Distribution")
plt.xlabel("Reconstruction Error")
plt.ylabel("Frequency")
plt.show()

# Bar chart for comparison
comparison.plot(kind='bar', figsize=(10, 5), title="Model Performance Comparison (Autoencoder vs Isolation Forest)")
plt.grid(True)
plt.show()
