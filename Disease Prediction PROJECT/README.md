### **README.md**  

```markdown
# Diabetes Detection Project  

This project uses Logistic Regression, a supervised machine learning algorithm, to predict whether an individual is diabetic or non-diabetic based on medical data. The model processes input features, trains on labeled data, and provides an interactive predictive system.  

---

## **Features**  
The model utilizes the following features for predictions:  
- **Pregnancies**: Number of pregnancies  
- **Glucose**: Plasma glucose concentration  
- **BloodPressure**: Diastolic blood pressure  
- **SkinThickness**: Triceps skinfold thickness  
- **Insulin**: 2-hour serum insulin level  
- **BMI**: Body Mass Index  
- **DiabetesPedigreeFunction**: Family history of diabetes (score)  
- **Age**: Individual's age  

The target variable (`Outcome`):  
- **1**: Diabetic  
- **0**: Non-Diabetic  

---

## **How It Works**  

### **Model Training and Testing**  
1. **Data Preprocessing**:  
   - Load the data from `medical_data.csv`.  
   - Split the data into features (`X`) and target (`y`).  
   - Scale the feature data using `StandardScaler`.  

2. **Model Development**:  
   - Train a Logistic Regression model using the training dataset.  
   - Evaluate the model's accuracy on the test dataset.  

3. **Interactive Prediction System**:  
   - Users can input their medical data interactively via a command-line interface.  
   - The model processes input and predicts whether the user is diabetic or non-diabetic.  

---

## **Installation**  

### **Prerequisites**  
- Python 3.x  

### **Setup**  
1. Clone this repository:  
   ```bash
   git clone https://github.com/yourusername/diabetes-detection.git
   cd diabetes-detection
   ```  

2. Create a virtual environment (optional but recommended):  
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   source venv/bin/activate  # On macOS/Linux
   ```  

3. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```  

---

## **Usage**  

### **Run the Model**  
1. Place your dataset (`medical_data.csv`) in the specified folder.  
2. Execute the script:  
   ```bash
   python model.py
   ```  

3. Follow the interactive prompt for predictions.  

### **Example Input**  
```json
{
    "Pregnancies": 3,
    "Glucose": 120,
    "BloodPressure": 80,
    "SkinThickness": 25,
    "Insulin": 130,
    "BMI": 28.5,
    "DiabetesPedigreeFunction": 0.6,
    "Age": 35
}
```  

### **Example Output**  
```
The accuracy of this model is 0.85

BELOW IS THE CLASSIFICATION REPORT OF THIS MODEL:

              precision    recall  f1-score   support
           0       0.90      0.85      0.87        40
           1       0.82      0.88      0.85        30
           
Do you want to check if you are diabetic or not? Y/N
Input data in the order as follows:
Pregnancies: 3
Glucose: 120
BloodPressure: 80
SkinThickness: 25
Insulin: 130
BMI: 28.5
DiabetesPedigreeFunction: 0.6
Age: 35
NOT diabetic
```  

---

## **File Descriptions**  
- `model.py`: Core script for training the model, evaluating it, and running predictions.  
- `medical_data.csv`: Dataset used for training and testing the model.  
- `requirements.txt`: List of Python dependencies required for the project.  

---

## **Technologies Used**  
- Python  
- Scikit-Learn  
- Pandas  
- Matplotlib  
- Seaborn  

---

## **Acknowledgments**  
- Dataset: [Pima Indians Diabetes Database](https://www.kaggle.com/uciml/pima-indians-diabetes-database)  

---

## **Contact**  
For questions, reach out to:  
- **Name**: Jeffery Ocran  
- **Email**: ocranjeffery@gmail.com  
- **GitHub**: [3kns](https://github.com/3kns)  

