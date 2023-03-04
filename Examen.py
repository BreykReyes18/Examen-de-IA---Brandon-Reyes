#Examen de Inteligencia Artificial - Brandon Isaac Cruz Reyes

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np

# Define los datos de entrada y salida
Kilometros = np.array([76, 100, 42, 172, 185, 57, 91, 98, 4, 164, 87, 49, 12, 91, 102, 78, 2, 49, 73], dtype=int)
Metros = np.array([76000, 100000, 42000, 172000, 185000, 57000, 91000, 98000, 4000, 164000, 87000, 49000, 12000, 91000, 102000, 78000, 2000, 49000, 73000], dtype=int)

# Crea el modelo de la red neuronal
model = keras.Sequential([
    keras.layers.Dense(19, activation='relu', input_shape=[1]),
    keras.layers.Dense(1)
])

# Compila el modelo
model.compile(loss='mean_squared_error', optimizer=tf.optimizers.Adam(0.98))

# Entrena el modelo con los datos de entrada y salida
entrenar = model.fit(Kilometros, Metros, epochs=1600, verbose=False)

print(model.predict([120]))

#Grafica del proceso de aprendizaje de la red neuronal
fig, ax=plt.subplots()
ax.set_title("Grafica del aprendizaje de la red neuronal")
ax.set_ylabel("Magnitud")
ax.set_xlabel("Numero de Epocas")
ax.xaxis.grid(True)
ax.yaxis.grid(True)
ax.plot(entrenar.history['loss'],color='blue', linestyle='-')
plt.show()
