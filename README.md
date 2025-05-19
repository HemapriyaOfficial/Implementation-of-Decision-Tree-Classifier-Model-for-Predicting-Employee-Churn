# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.import pandas module and import the required data set.

2.Find the null values and count them.

3.Count number of left values.

4.From sklearn import LabelEncoder to convert string values to numerical values.

5.From sklearn.model_selection import train_test_split.

6.Assign the train dataset and test dataset.

7.From sklearn.tree import DecisionTreeClassifier.

8.Use criteria as entropy.

9.From sklearn import metrics.

10.Find the accuracy of our model and predict the require values.


## Program:
```

Developed by: HEMAPRIYA K
RegisterNumber:  212223040066

import pandas as pd

data = pd.read_csv("Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data["salary"] = le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project", "average_montly_hours",
"time_spend_company", "Work_accident","promotion_last_5years","salary"]]
x.head()

y = data["left"]

from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn. tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt. predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)

accuracy
dt.predict([[0.5,0.8,9,260, 6,0,1,2]])

```

## Output:
![435883921-dfc48f3d-77b0-4845-989e-af4a0408c65a](https://github.com/user-attachments/assets/96f4a57d-079d-41bb-aa0a-a3ea7966e5b2)


![435883971-6f500b5e-3c6c-4533-b667-99ec3235be9c](https://github.com/user-attachments/assets/46b4b018-26fd-4253-a070-8b2e168a1902)

![435884002-1eab8755-d171-4078-933b-2e395dabeb87](https://github.com/user-attachments/assets/5fd0d18e-987e-4072-ab42-0c88f75cb3fa)


![435884333-29d2a3a5-b101-4476-8302-a907b5a0b3b0](https://github.com/user-attachments/assets/6ead3242-24b6-424f-a61e-e7d79fad85a9)


![435884705-5033fe2d-e2ad-4907-829e-3197d08aa4bd](https://github.com/user-attachments/assets/81ae278f-d459-461d-93fe-1daeac5f3301)


![435884382-eb5fdbac-1311-4a3a-98fe-5db11209ea95](https://github.com/user-attachments/assets/713b688c-6e0b-4e21-8c60-49a477bba0ec)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
