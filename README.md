# 🏠 Boston Housing Price Prediction

This repository contains a Jupyter Notebook (`BostonHousing.ipynb`) that builds a machine learning model to predict housing prices using the classic Boston Housing dataset. The notebook walks through data preprocessing, visualization, model training, and evaluation.

## 📘 Overview

The Boston Housing dataset contains information about various features of houses in Boston suburbs, such as number of rooms, crime rate, and property tax rate. The goal is to predict the **median house value** based on these features.

## 🧰 Technologies Used

- Python 3.x
- Jupyter Notebook
- scikit-learn
- pandas
- NumPy
- Matplotlib / Seaborn

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/BostonHousing.git
cd BostonHousing
2. Install dependencies
bash
Copy
Edit
pip install pandas numpy scikit-learn matplotlib seaborn
3. Launch the notebook
bash
Copy
Edit
jupyter notebook BostonHousing.ipynb
🔍 Workflow
Load the dataset (from sklearn.datasets)

Explore and visualize the data

Preprocess the features (normalization, train/test split)

Train regression models (e.g., Linear Regression, Decision Tree, Random Forest)

Evaluate model performance using metrics like MSE and R² score

📊 Example Output
yaml
Copy
Edit
Model: Random Forest Regressor
R² Score: 0.89
Mean Squared Error: 12.3
📈 Visualizations
Correlation heatmap

Feature importance plots

Actual vs. Predicted value scatter plot

📄 License
This project is licensed under the MIT License.
