import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from haversine import haversine, Unit
import os

data_list = []
rootdir = ''

for subdir, dirs, files in os.walk(rootdir):
    for file_path in files:
        if file_path[:-3]=='csv':
            data = pd.read_csv(file_path)

            data['displacement'] = data.apply(lambda row: haversine((row['latitude'], row['longitude']),
                                                                    (data['latitude'].shift(-1)[row.name], data['longitude'].shift(-1)[row.name]
                                                                    if row.name < len(data) - 1 else np.nan), 
                                                                    Unit=Unit.METERS), axis=1) 

            data = data.dropna(subset=['displacement'])
            data_list.append(data)

data = pd.concat(data_list, ignore_index=True)

X = data[['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'compass']]
y = data['displacement']

split_idx = int(0.8 * len(data))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Root Mean Squared Error: {rmse}")
