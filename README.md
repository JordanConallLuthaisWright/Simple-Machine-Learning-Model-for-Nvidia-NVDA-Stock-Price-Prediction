# Simple Stock Price Prediction ML Model for Nvidia (NVDA) [Feb 2025]

***(Dataset taken from Yahoo Finance)***

## **Project Overview**
This project focuses on **predicting Nvidia (NVDA) stock price movements** using **machine learning models**. By analyzing **historical stock market data**, the model aims to classify whether the stock price will **increase or decrease** the next day.

By leveraging **data preprocessing, feature engineering, backtesting, and predictive modeling**, this project explores the potential of **AI in financial forecasting**. Key tasks include **creating moving averages, trend indicators, and testing multiple classification models**, ensuring a structured approach to financial data analysis.

The structured datasets support **investment decision-making, algorithmic trading, and risk management**, demonstrating practical **machine learning applications** in financial markets.

The goal is to **train and evaluate a model that predicts** whether Nvidia's stock price will rise or fall the following day.

---

## **Project Aim**  
This project aims to **apply machine learning techniques to forecast short-term Nvidia stock price movements**, enhancing **data-driven investment strategies**.

The transformation process prioritizes:
- **Feature engineering** to extract meaningful market signals.
- **Model training and evaluation** using historical price trends.
- **Backtesting** to assess the model's performance over different time periods.

---

## **Tech Stack & Skills Demonstrated**
- **Python**: Data processing and predictive modeling.
- **Pandas & NumPy**: Data manipulation and statistical analysis.
- **Scikit-learn**: Machine learning for classification models.
- **Matplotlib & Seaborn**: Data visualization of stock trends.
- **Backtesting & Model Evaluation**: Assessing model accuracy over time.

## **Technical Skills Demonstrated**  
- **Financial Data Preprocessing**: Cleaning and structuring historical stock data.  
- **Feature Engineering**: Creating **rolling averages and trend indicators**.  
- **Machine Learning Classification**: Training **Random Forest models**.  
- **Backtesting**: Evaluating model performance over different time periods.  
- **Performance Analysis**: Using **precision scores and visualization** to assess accuracy.  

---

## **Files in This Repository**  

| File | Description |
|------|------------|
| `Nvidia_Stock_Prediction.ipynb` | Jupyter Notebook with data preprocessing, feature engineering, and model training. |
| `README.md` | Project documentation. |

---

## **How to Run This Project**  

1. **Clone the repository**  
   ```bash
   git clone https://github.com/JordanConallLuthaisWright/Simple-Machine-Learning-Model-for-Nvidia-NVDA-Stock-Price-Prediction.git
2. **Navigate to the project directory**
   ```bash
   cd Simple Machine Learning Model for Nvidia (NVDA) Stock Price Prediction
3. **Install dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
4. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
5. Open and execute the cells, "Simple Machine Learning Model for Nvidia (NVDA) Stock Price Prediction.ipynb"

---

## **Business Context & Problem Statement**  
Nvidia’s stock price has **shown strong growth trends**, but also **high volatility**, making it difficult to predict short-term movements. This project explores whether **machine learning models** can capture **price trends and momentum indicators** to improve forecasting.

### **Key Challenges:**  
- **Financial markets are highly unpredictable.**  
- **Machine learning models need meaningful historical features** to make informed predictions.  
- **Backtesting is essential** to ensure the model performs reliably over unseen data.  

---

## **Methodology & Technical Steps**  

### **1. Data Preprocessing & Cleaning**  
- **Extracted historical stock prices** from Yahoo Finance.  
- **Created rolling averages** to capture long-term trends.  

### **2. Feature Engineering**  
- **Moving Averages** (2, 5, 60, 250, 1000-day periods).  
- **Trend Indicators**: Count of previous **up-days to measure momentum**.  
- **Binary Target Creation**: `1` if the price increases, `0` otherwise.  

### **3. Model Training & Evaluation**  
- **Random Forest Classifier** as the **baseline model**.  
- **Tuned hyperparameters** for better performance.  
- **Evaluated with precision scores and classification metrics**.  

### **4. Backtesting Approach**  
- **Tested model over multiple historical periods**.  
- **Evaluated whether predictions aligned with actual stock movements**.  

---

## **Key Takeaways & Impact**  
- **Stock price prediction is complex**, but **feature engineering improves model accuracy**.  
- **Early model performance achieved ~51.7% precision** (near random).  
- **Feature Engineering boosted precision to 56.7%**, highlighting the importance of **trend signals**.  
- **Backtesting provided real-world validation** of model effectiveness.  

---

## **Future Enhancements**  

### **1. Deep Learning Models**  
- Implement **LSTMs or GRUs** for better **time-series forecasting**.  

### **2. Incorporate Macroeconomic Factors**  
- Include **earnings reports, Federal Reserve decisions, and global economic data**.  

### **3. Test Alternative Machine Learning Models**  
- Compare **XGBoost, SVM, and Logistic Regression** to assess model performance.  

### **4. Live Data Testing**  
- Apply predictions on **real-time stock data** for validation.  

---

## **Business Impact & Insights**  
This project demonstrates **how AI can assist in financial decision-making**. By refining prediction models, traders and investors can:  

**Enhance algorithmic trading strategies.**  
**Identify trends more effectively.**  
**Test machine learning models on real market data.**  

Although **stock prices are influenced by numerous external factors**, this project shows that **historical price trends and feature engineering can provide valuable insights into price direction**.  

---

## **Contact & Contributions**  
Feel free to explore and contribute! If you have any suggestions, reach out or submit a pull request.  

- **Email**: [jordan.c.l.wright@gmail.com](mailto:jordan.c.l.wright@gmail.com)  

---

### **Author:** Jordan  
[GitHub Profile](https://github.com/JordanConallLuthaisWright)  

