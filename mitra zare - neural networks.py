

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
import matplotlib.pyplot as plt

data = {
    "outlook": ["sunny", "sunny", "overcast", "rainy", "rainy", "rainy", "overcast", "sunny", "sunny", "rainy", "sunny", "overcast", "overcast", "rainy"],
    "temperature": [85, 80, 83, 70, 68, 65, 64, 72, 69, 75, 75, 72, 81, 71],
    "humidity": [85, 90, 86, 96, 80, 70, 65, 95, 70, 80, 70, 90, 75, 91],
    "wind": ["FALSE", "TRUE", "FALSE", "FALSE", "FALSE", "TRUE", "TRUE", "FALSE", "FALSE", "FALSE", "TRUE", "TRUE", "FALSE", "TRUE"],
    "play": ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no"]
}

df = pd.DataFrame(data)

label_encoder = LabelEncoder()
df['outlook'] = label_encoder.fit_transform(df['outlook'])
df['wind'] = label_encoder.fit_transform(df['wind'])
df['play'] = label_encoder.fit_transform(df['play'])  

scaler = StandardScaler()
df[['temperature', 'humidity']] = scaler.fit_transform(df[['temperature', 'humidity']])

X = df.drop('play', axis=1)
y = df['play']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=1, validation_data=(X_val, y_val))




