# elevatelabs3
# 🏡 House Price Prediction using Linear Regression  

## 📌 Project Overview  
This project predicts **house prices** based on various features using a **Linear Regression model**. We used the `Housing.csv` dataset, performed preprocessing, trained the model, and evaluated it with different metrics. The project demonstrates how regression can be applied to real-world datasets.  

---

## ⚙️ Steps Followed  

### **1. Importing Libraries**  
- Loaded required libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`.  

### **2. Loading Dataset**  
- Loaded the dataset (`Housing.csv`) using **pandas**.  
- Explored data using `.head()`, `.info()`, and `.describe()`.  

### **3. Data Preprocessing**  
- Handled categorical variables using **one-hot encoding** (`pd.get_dummies`).  
- Defined **features (X)** and **target (y)** (`price`).  

### **4. Train-Test Split**  
- Split the dataset into training and testing sets (`80-20`) using `train_test_split`.  

### **5. Model Training**  
- Trained a **Linear Regression model** from `sklearn.linear_model`.  

### **6. Predictions**  
- Made predictions on the test data using `.predict()`.  

### **7. Model Evaluation**  
- Evaluated performance using:  
  - **MAE (Mean Absolute Error)**  
  - **MSE (Mean Squared Error)**  
  - **RMSE (Root Mean Squared Error)**  
  - **R² (Coefficient of Determination)**  

### **8. Coefficient Interpretation**  
- Extracted **intercept** and **coefficients** to understand the effect of each feature on house prices.  

### **9. Visualization**  
- Plotted **Actual vs Predicted prices** with a scatter plot.  
- Added a regression line to visualize performance.  

---

## 📊 Results  
- Model performance metrics were displayed (MAE, MSE, RMSE, R²).  
- The scatter plot showed how well predictions matched actual values.  

---

## 🛠️ Tools & Libraries Used  
- **Python**  
- **Pandas** – Data handling  
- **NumPy** – Numerical operations  
- **Matplotlib & Seaborn** – Data visualization  
- **Scikit-learn** – Model building & evaluation  

---

## 🚀 How to Run  
1. Clone the repository  
   ```bash
   git clone https://github.com/your-username/house-price-prediction.git
   cd house-price-prediction
   ```
2. Install required libraries  
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook / Python script.  

---

## 📌 Future Improvements  
- Try **advanced models** like Ridge, Lasso, Random Forest, or XGBoost.  
- Perform **feature scaling** and **feature selection**.  
- Deploy the model using **Flask/Streamlit** for interactive predictions.  

---

✨ **This project is a simple but solid demonstration of Linear Regression for predicting real-world values like house prices.**


