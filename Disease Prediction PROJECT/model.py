import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report

data = pd.read_csv('Disease Prediction PROJECT/medical_data.csv')

x = data.drop(['Outcome'], axis= 1)
y = data['Outcome']

sc = StandardScaler()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, stratify= y, random_state= 1)

xv_train = sc.fit_transform(x_train)
xv_test = sc.transform(x_test)

classifier = SVC(kernel= 'linear')

classifier.fit(xv_train, y_train)
pred_classifier = classifier.predict(xv_test)

score = classifier.score(xv_test, y_test)

print(f'The accuracy of this model is {score}\n\n\n')
print(f'BELOW IS THE CLASSIFICATION REPORT OF THIS MODEL; \n\n{classification_report(pred_classifier, y_test)}')

while True:
    prompt = input('Do you want to check if you are diabetic or not? Y/N  ')
    prompt = prompt.lower()
    if prompt == 'y':
        print('\nInput data in the order as follows;')
        dict = {'Pregnancies': '', 'Glucose': '', 'BloodPressure': '', 'SkinThickness': '', 'Insulin': '', 'BMI': '', 'DiabetesPedigreeFunction': '', 'Age': ''}
        for i in dict:
            dict[i] = input(f'{i}:')

        input_data = pd.DataFrame([dict])

        prediction = classifier.predict(input_data)

        if prediction == 0:
            print('NOT diabetic')
        elif prediction == 1:
             print('Diabetic, please visit your doctor.')
    elif prompt == 'n':
        print('Sure. Any other time, I will be available')
        break
    else:
        print('Your input was not understood')