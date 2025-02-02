# **Optimizing Bank Marketing Campaigns with AI**
### **Predicting Customer Subscription to a Term Deposit in a Portuguese Bank**

---

## 🎯 Business Impact: Why is this Useful?

This model allows the bank to:
1. **Reduce Marketing Costs** → The bank can avoid calling uninterested customers.
2. **Improve Efficiency** → Focus on high-potential leads, increasing campaign success rates.
3. **Enhance Customer Experience** → Avoid spamming uninterested customers.
4. **Increase Revenue** → More term deposit subscriptions lead to better bank profitability.

---

## **📌 Background: Why Train the Model?**
The Portuguese banking institution has been running **direct phone-based marketing campaigns** to encourage customers to **subscribe to term deposits**. However, these campaigns involve **multiple calls per customer**, leading to high operational costs and wasted resources on uninterested clients.

To optimize these efforts, the bank wants to **train a machine learning model** that can predict which customers are **most likely to subscribe to a term deposit**. This prediction will allow the bank to:
- Focus marketing efforts on **highly interested customers**.
- Reduce **unnecessary phone calls** and operational costs.
- Improve **conversion rates** and overall campaign efficiency.

The dataset contains **detailed information about customers and their interactions with previous marketing campaigns**, including **demographics, past interactions, and loan history**.

Dataset: https://archive.ics.uci.edu/dataset/222/bank+marketing

---

## **📊 Understanding the Dataset**
The dataset consists of **41,188 customer records** with **20 features**, collected between **May 2008 and November 2010**. The target variable **(`y`)** is **binary**:
- **"yes"** → The customer subscribed to a term deposit.
- **"no"** → The customer did not subscribe.

### **🔹 Key Features in the Dataset**
1. **Customer Demographics**
   - `age`: Age of the customer.
   - `job`: Type of job (e.g., "admin", "blue-collar", "student").
   - `marital`: Marital status (e.g., "married", "single").
   - `education`: Level of education (e.g., "university degree", "high school").

2. **Financial Information**
   - `balance`: Customer’s average yearly balance (in euros).
   - `default`: Whether the customer has **credit in default**.
   - `housing`: Whether the customer has a **housing loan**.
   - `loan`: Whether the customer has a **personal loan**.

3. **Marketing Campaign Interactions**
   - `contact`: Type of contact used (cellular vs. telephone).
   - `day_of_week` & `month`: When the customer was last contacted.
   - `duration`: Last call duration (**⚠️ should be discarded for realistic predictions** as it’s known after the call).
   - `campaign`: Number of contacts in the current campaign.
   - `pdays`: Days since the customer was last contacted (**-1 means no prior contact**).
   - `previous`: Number of previous contacts.
   - `poutcome`: Outcome of the previous campaign (e.g., "failure", "success").

---

## **🛠️ Code**

### **Step 1: Load the Dataset**
```python
import pandas as pd

# Load the dataset
df = pd.read_csv("bank-additional-full.csv", sep=";")  # Using semicolon as separator
```

- The bank has recorded customer responses from past campaigns.
- The dataset is loaded for preprocessing and analysis.

### **Step 2: Data Preprocessing**
Before training a model, we need to clean and transform the data.
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Identify categorical and numerical features
categorical_features = ["job", "marital", "education", "contact", "month", "poutcome"]
numerical_features = ["age", "balance", "campaign", "pdays", "previous"]

# Drop 'duration' since it's not available before a call is made
df = df.drop(columns=["duration"])

# Encode categorical and numerical data
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),  # Scale numerical values
        ("cat", OneHotEncoder(), categorical_features)  # Convert categorical values
    ]
)

# Encode target variable
le = LabelEncoder()
df["y"] = le.fit_transform(df["y"])  # Convert 'yes'/'no' to 1/0

# Split into train and test sets
X = df.drop(columns=["y"])
y = df["y"]
X_train, X_test, y_train, y_test = train_test_split(preprocessor.fit_transform(X), y, test_size=0.2, random_state=42)
```

Why do this?
- **StandardScaler** ensures that numeric values are normalized.
- **OneHotEncoder** converts categorical values into **machine-readable format**.
- **Drop ‘duration’** because **it is only available after the call**.
- **Train-test split** ensures we test our model on unseen customers.

### **Step 3: Train a Deep Learning Model**
The bank wants to use a neural network to capture complex relationships between customer data and subscription likelihood.
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Define a neural network model
model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation="relu"),
    Dropout(0.5),  # Prevents overfitting
    Dense(32, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")  # Outputs probability (0 to 1)
])

# Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

```

The deep learning model used in this script is a **Multi-Layer Perceptron (MLP)**, a type of Artificial Neural Network (ANN). It consists of fully connected (dense) layers and is designed for binary classification.

#### 📌 Why Use This Deep Learning Model for This Dataset?

A Multi-Layer Perceptron (MLP) is used because:

1. Handles Both Categorical & Numerical Data
  - The dataset includes both categorical (e.g., job, marital status) and numerical (e.g., balance, age) features.
  - The preprocessing pipeline encodes categorical variables and normalizes numerical values, making it well-suited for an MLP.

2. Captures Complex Patterns
  - Bank marketing data has non-linear relationships between features.
  - MLP can detect interactions between variables (e.g., how age + job type + previous campaign success influences customer behavior).

3. Better Than Logistic Regression for Large Datasets
  - Logistic Regression works well for simple binary classification but struggles with high-dimensional data.
  - MLP can learn complex decision boundaries from features.

4. Dropout Layers Help Prevent Overfitting
  - Overfitting is a common issue in marketing datasets.
  - Dropout (0.5) ensures the model generalizes well to unseen data.


### **Step 3: Train a Deep Learning Model**
The bank wants to use a neural network to capture complex relationships between customer data and subscription likelihood.
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Define a neural network model
model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation="relu"),
    Dropout(0.5),  # Prevents overfitting
    Dense(32, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")  # Outputs probability (0 to 1)
])

# Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
```

### **Step 4: Making Predictions on New Customers**
The trained model can now be used to predict if new customers are likely to subscribe.
```python
# Predict on test data
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int).flatten()  # Convert probability to binary class
```
- If `y_pred = 1`, the customer is likely to subscribe.
- If `y_pred = 0`, the customer is unlikely to subscribe.

### **Step 5: Evaluating the Model’s Performance**
How well does the model perform in predicting customer subscriptions?
```python
from sklearn.metrics import precision_score, recall_score, f1_score

# Evaluate model performance
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 Score: {f1:.3f}")
```

**✅ Precision** → How many of the predicted "yes" were actually correct?

**✅ Recall** → How many actual "yes" were correctly identified?

**✅ F1 Score** → A balance between Precision and Recall.

### 📌 How can we integrated the model we developed?
We can create a real-time prediction for Bank Marketing Campaigns. Here's how I imagine it would look like.

#### 🏆 Scenario  
A **surveyor** is using a **mobile application** during a **field campaign** to gather customer information. 📱  

- **📝 Data Collection**  
  - The surveyor inputs the following details into the app for a new customer, **John**:  
    ✅ **Age:** 35  
    ✅ **Marital Status:** Married  
    ✅ **Job:** Blue-collar  
    ✅ **Housing Loan:** No  
    ✅ **Personal Loan:** No  
    ✅ **Contact Method:** Cellular  
    ✅ **Outcome of Previous Campaign:** Success  

- **⚡ Real-Time Prediction**  
  - 📤 Upon submitting John's information, the app **sends a request** to the **Django-based API** hosting the **predictive model**.  
  - 💡 The API **processes the data** and returns a prediction:  
  - 📊 **Probability of Subscription:** **78%**  

- **🎯 Actionable Insight**  
  - 🔔 The app displays a notification:  
  - **"High likelihood of subscription. Prioritize follow-up with this customer."**
  - 💬 The surveyor, **armed with this insight**, can tailor the conversation to **increase the chances of conversion**.  

🚀 **With this AI-driven approach, marketing efforts become more efficient, improving success rates while reducing wasted resources!**  

## 🚀 Next Steps
- **Fine-tune the Model**: Try Random Forest or XGBoost for comparison.
- **Deploy the Model**: Build a web app for real-time customer prediction.
- **Analyze Feature Importance**: Determine which factors influence decisions the most.

P.S. You can run this code on [Google Colab](https://colab.research.google.com/drive/15SOzhBogFUWrBAXZuUdhAG09KoaHlOY9?usp=sharing).
  
