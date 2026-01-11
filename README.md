# 🌤️ Hybrid EfficientNet-B0 + BiLSTM Framework for Solar Irradiance Forecasting Using Infrared Sky Imaging

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://www.python.org/) [![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12-orange?logo=tensorflow)](https://www.tensorflow.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE) ![Last Update](https://img.shields.io/badge/Last%20Update-Nov%202025-brightgreen)

> A deep learning-based hybrid framework combining **EfficientNet-B0 (CNN)** and **BiLSTM (RNN)** for accurate **short-term solar irradiance forecasting** using **infrared sky imaging** and real-time visualization.

---

## ☀️ Project Overview

Accurate solar irradiance forecasting is vital for **solar power management, smart grids, and renewable energy optimization**.
This project proposes a **Hybrid EfficientNet-B0 + BiLSTM model** that integrates **spatial feature extraction** with **temporal sequence forecasting** to deliver precise real-time irradiance predictions.

### 🔍 Key Highlights

* 🖼️ **EfficientNet-B0** for spatial feature extraction (clouds, texture, brightness)
* ⏳ **BiLSTM** for temporal dependency learning and forecasting
* 🌈 **Infrared (IR) image preprocessing** with colormap and interpolation
* ⚡ **Flask-based web dashboard** for real-time irradiance visualization

---

## 🧠 Methodology

### 1️⃣ Data Acquisition

* **Dataset:** [GIRASOL Dataset](https://doi.org/10.1016/j.dib.2021.106914)
* **Data Includes:**

  * Infrared (IR) sky images captured every **15 seconds**
  * Pyranometer-measured irradiance values (W/m²)
  * Metadata: timestamps, sun position, ambient temperature, humidity
  * Duration: December 2017 – January 2019

---

### 2️⃣ Preprocessing Pipeline

To enhance raw infrared images for better feature extraction:

* 🔹 **Normalization:** Scales pixel values to [0,1]
* 🔹 **Bicubic Interpolation:** Upscales IR images for smoother resolution
* 🔹 **OpenCV JET Colormap:** Converts grayscale IR images to RGB
* 🔹 **Timestamp Alignment:** Synchronizes image and irradiance pairs

> Output: Enhanced **224×224×3 RGB IR Images** ready for training

---

### 3️⃣ Model Architecture

#### ⚙️ **EfficientNet-B0 (CNN) – Nowcasting**

* Extracts spatial features from enhanced IR images
* Predicts **current solar irradiance**
* Lightweight and efficient model with compound scaling

#### 🔁 **BiLSTM (Bidirectional LSTM) – Forecasting**

* Takes a sequence of **20 CNN outputs (past 5 min)**
* Forecasts **next 1-minute irradiance** at 15s intervals
* Captures both forward and backward temporal dependencies

#### 🧩 **Hybrid Framework Workflow**

```
IR Image → Preprocessing → EfficientNet-B0 → Sequence Generator → BiLSTM → Forecasted Irradiance
```

---

### 4️⃣ Evaluation Metrics

To assess forecasting accuracy:

* **MAE (Mean Absolute Error)**
* **MSE (Mean Squared Error)**
* **RMSE (Root Mean Squared Error)**
* **R² Score (Coefficient of Determination)**

> ✅ Achieved ~29% improvement in RMSE compared to baseline CNN/LSTM models.

---

## 💻 Project Structure

```
Hybrid-EfficientNet-B0-BiLSTM-Solar-Forecasting/
│
├── data/                # GIRASOL dataset & processed images
├── models/              # Saved model weights (.h5)
├── notebooks/           # Jupyter notebooks for training/evaluation
├── app/                 # Flask web application
├── scripts/             # Preprocessing and dataset handling scripts
├── static/ & templates/ # Web dashboard assets (HTML, CSS, JS)
├── README.md            # Project documentation
└── requirements.txt     # Dependencies
```

---

## 🌐 Web Dashboard

A real-time **Flask + Chart.js dashboard** for interactive visualization:

* 📤 Upload infrared images or test dataset
* ⚙️ Predict current and next-minute irradiance values
* 📊 Visualize real-time **Actual vs Predicted** irradiance graphs

> Demonstrates real-world usability for **solar plant operators and energy systems**.

---

## 📊 Results

| Metric       | Value     | Description                          |
| ------------ | --------- | ------------------------------------ |
| **MAE**      | *Low*     | Small average prediction error       |
| **MSE**      | *Low*     | Penalizes larger deviations          |
| **RMSE**     | ↓ **29%** | Improved accuracy vs baseline        |
| **R² Score** | *High*    | Strong correlation with ground truth |

🖼️ **Model Visualization:**

* Predicted vs Actual Irradiance Curve
* Error Distribution Graph

---

## 🧰 Tech Stack

* 🐍 **Python 3.9+**
* 🧠 **TensorFlow / Keras**
* 🔢 **NumPy, Pandas, Scikit-learn**
* 🎨 **OpenCV, Matplotlib, Seaborn**
* 🌐 **Flask, Chart.js (Web Interface)**

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/Anand-b-patil/Hybrid-EfficientNet-B0-BiLSTM-Solar-Forecasting.git
cd Hybrid-EfficientNet-B0-BiLSTM-Solar-Forecasting

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Run the Application

```bash
# Step 1: Preprocess IR images
python scripts/preprocess.py

# Step 2: Train EfficientNet-B0 + BiLSTM model
python notebooks/train_hybrid_model.py

# Step 3: Launch Flask Web App
python python -m uvicorn Fast_api_app:app --reload --host 127.0.0.1 --port 5000
```

> The dashboard will be available at: **[http://localhost:5000](http://localhost:5000)**

---

## 📚 References

* 📜 [A Hybrid CNN-LSTM Framework and Infrared Image Processing for Solar Irradiance Forecasting](https://ieeexplore.ieee.org/document/10906220)
* 🌍 [GIRASOL Dataset – MDPI Sensors Journal](https://doi.org/10.1016/j.dib.2021.106914)

---

## 🌞 Applications

* ⚡ Real-time solar power plant monitoring
* 🏙️ Smart grid energy balancing
* 🏡 Rural microgrid management
* 🔆 Solar panel tracking and optimization
* 🌦️ Weather prediction and atmospheric research

---

## 🤝 Contributing

Contributions and ideas are welcome!
Feel free to **fork the repository**, improve features, or suggest enhancements via **pull requests** 🌟

---

## 🧾 License

This project is licensed under the **MIT License** 📝

---

## 👨‍💻 Author

**Anand Bhimagouda Patil**
📧 [ap6272440@gmail.com](mailto:ap6272440@gmail.com)
🔗 [GitHub](https://github.com/Anand-b-patil) | [LinkedIn](https://linkedin.com/in/anand_b_patil)

