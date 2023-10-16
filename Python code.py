
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


data = {
    'Date': pd.date_range(start='2023-01-01', periods=365),
    'Temperature': np.random.randint(0, 100, size=365)
}

df = pd.DataFrame(data)


X = df[['Date']].values.reshape(-1, 1)
y = df['Temperature']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_pred, color='red')
plt.title('Temperature Prediction')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.show()
