#importing essential libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

#reading medical data
data = pd.read_csv('Disease Prediction PROJECT/medical_data.csv')

# separating the data to be trained and tested from the outcome
x = data.drop(['Outcome'], axis= 1)
y = data['Outcome']
# 1 is DIABETIC and 0 is NON-DIABETIC

# initialisng the standardscaler function
sc = StandardScaler()
x = sc.fit_transform(x)

#splitting the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, stratify= y, random_state= 1)


# initialising the model
lr = LogisticRegression()

#training the model
lr.fit(x_train, y_train)

# predicting using the test data
pred_lr = lr.predict(x_test)

# analysing how our accurate our model is
pred_accuracy_score = lr.score(x_test, y_test)

print(f'The accuracy of this model is {pred_accuracy_score}\n\n\n')
print(f'BELOW IS THE CLASSIFICATION REPORT OF THIS MODEL; \n\n{classification_report(pred_lr, y_test)}')


# PREDICTIVE SYSTEM
while True:
    prompt = input('Do you want to check if you are diabetic or not? Y/N  ')
    # converting prompt into lower case
    prompt = prompt.lower()

    #setting conditions
    if prompt == 'y':
        print('\nInput data in the order as follows;')
        dict = {'Pregnancies': '', 'Glucose': '', 'BloodPressure': '', 'SkinThickness': '', 'Insulin': '', 'BMI': '', 'DiabetesPedigreeFunction': '', 'Age': ''}
        # giving values to the dictionary keys
        for i in dict:
            dict[i] = float(input(f'{i}:'))
        # change the dictionary into a dataframe for predicition
        input_data = pd.DataFrame([dict])

    

        std_data = sc.transform(input_data)
        

        # predicting
        prediction = lr.predict(std_data)
        print(prediction)

        #output conditions
        if prediction == 0:
            print('NOT diabetic')
        elif prediction == 1:
             print('Diabetic, please visit your doctor.')
    elif prompt == 'n':
        print('Sure. Any other time, I will be available')
        break
    else:
        print('Your input was not understood')