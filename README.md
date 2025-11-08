# ⚙️ Bearing Fault Detection using Autoencoder and Isolation Forest

> **Unsupervised Anomaly Detection for Turbine Bearing Health Monitoring**  
> This project compares **Autoencoder Neural Networks** and **Isolation Forest** techniques for detecting bearing faults using real-time sensor data such as vibration, temperature, pressure, and power output.  

---

## 📖 Overview

Bearings are essential components in turbines and rotating machines. A minor fault can lead to serious mechanical failure or downtime.  
This project demonstrates how **unsupervised machine learning** can be applied for **early fault detection** in such systems, even when labeled fault data is unavailable.

Two techniques are implemented and compared:

- 🧠 **Autoencoder (Deep Learning)** — Learns normal behavior and flags deviations based on reconstruction error.  
- 🌲 **Isolation Forest (Machine Learning)** — Detects anomalies by isolating outliers in feature space.

---

## 🎯 Objectives

- Detect early bearing faults in turbines and industrial equipment.  
- Compare **Autoencoder** and **Isolation Forest** performance on sensor data.  
- Provide a foundation for **predictive maintenance** and **SCADA integration**.

---

## ⚙️ Techniques Used

### 🧠 Autoencoder
- A neural network trained to **reconstruct input data** representing normal system behavior.  
- High reconstruction error → indicates potential fault or anomaly.  
- Works best with large datasets and multiple correlated features.

### 🌲 Isolation Forest
- A tree-based anomaly detection algorithm that isolates outliers by random partitioning.  
- Efficient for smaller datasets and interpretable.  
- Quick to train and implement.

---

## 📊 Dataset

The model uses time-series sensor data typically collected from turbines or industrial bearings:
- Vibration signals  
- Temperature readings  
- Power output  
- Oil pressure and flow rate  

Example datasets: **IMS Bearing Dataset**, or any **SCADA-based sensor logs**.

---

## 🧩 Model Comparison

| Method | Key Strengths | Limitations |
|--------|----------------|-------------|
| **Autoencoder** | Learns complex nonlinear relations, ideal for high-dimensional data | Needs more data & compute time |
| **Isolation Forest** | Fast, lightweight, easy to interpret | May miss subtle faults in complex datasets |

---

## 🧰 Technologies Used

- **Python 3.x**
- **NumPy** and **Pandas** – Data handling  
- **Matplotlib** and **Seaborn** – Visualization  
- **Scikit-learn** – Isolation Forest & metrics  
- **TensorFlow / Keras** – Autoencoder model  
- **StandardScaler** – Feature normalization  
- **Confusion Matrix & Classification Metrics** – Evaluation

---

## ⚡ Results Summary

- Both models successfully detect bearing anomalies.  
- **Autoencoder** achieves higher accuracy on complex, multi-parameter datasets.  
- **Isolation Forest** performs better for quick fault screening.  

---

## 🔍 Applications

- Predictive Maintenance in **Hydro Power Plants**  
- Fault Detection in **Rotating Machinery**  
- Integration with **SCADA Systems** for real-time monitoring  
- Early Warning & **Equipment Health Tracking**

---

## 🧾 Conclusion

This project highlights the capability of **unsupervised learning** in industrial fault detection.  
With real-time SCADA integration, it can help prevent costly turbine breakdowns and ensure continuous plant operation.

Future improvements may include:
- CNN/LSTM-based deep learning for time-series sequences  
- Live data streaming and dashboard visualization  
- Automated threshold tuning for anomaly detection

---

## 🧪 Example Evaluation Metrics

```python
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))
print("Accuracy:", accuracy_score(y_true, y_pred))
