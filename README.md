# 🌸 Iris Flower Classification using ANN

This project predicts the species of Iris flowers (**Iris-setosa**, **Iris-versicolor**, or **Iris-virginica**) using a deep learning **Artificial Neural Network (ANN)**.  
It includes:

- **Data Preprocessing** (Scaling & Class Balancing)
- **Model Training** using TensorFlow/Keras
- **Gradio Web App** for live predictions
- **Metrics Visualization** (Accuracy, Classification Report, Confusion Matrix)

---

## 📂 Project Structure
Iris-ANN-Project/
│
├── Iris.csv # Original dataset
├── Iris_scaled.csv # (Optional) Scaled dataset
├── app.py # Gradio web app for predictions
├── train_model.py # Script to train ANN and save model
├── iris_ann_model.h5 # Saved ANN model
├── iris_scaler.pkl # Saved StandardScaler
├── label_map.pkl # Label encoder mapping
├── requirements.txt # Dependencies
└── README.md # Project documentation


---

## 📊 Dataset

We are using the **Iris dataset**, which contains the following features:

| Feature          | Description |
|------------------|-------------|
| SepalLengthCm    | Sepal length in cm |
| SepalWidthCm     | Sepal width in cm |
| PetalLengthCm    | Petal length in cm |
| PetalWidthCm     | Petal width in cm |
| Species          | Flower species (target variable) |

### Example:
| SepalLength | SepalWidth | PetalLength | PetalWidth | Species |
|-------------|------------|-------------|------------|---------|
| 5.1         | 3.5        | 1.4         | 0.2        | Iris-setosa |

---

## ⚙️ Installation

### 1. **Clone Repository**
```bash
git clone https://github.com/your-username/Iris-ANN-Project.git
cd Iris-ANN-Project

🚀 Model Training

Run the train_model.py script to:

Scale the features

Balance the dataset using RandomOverSampler

Train the ANN

Save the trained model and scaler

python train_model.py

Outputs after training:

iris_ann_model.h5 → Trained ANN model

iris_scaler.pkl → Scaler for future predictions

label_map.pkl → Mapping for target labels

📈 Metrics & Visualization

After training, the script will display:

Test Accuracy

Classification Report

Confusion Matrix Heatmap

📌 Future Improvements

Deploy the Gradio app on Hugging Face Spaces or Streamlit Cloud.

Experiment with different ANN architectures or hyperparameter tuning.

Add cross-validation for better model evaluation.

🤝 Contributors
KAVINKUMAR T – Developer & Maintainer
