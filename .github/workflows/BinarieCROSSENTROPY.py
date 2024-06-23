import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Definir y cargar los datos (ejemplo de puerta lógica OR)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)

# Definir la arquitectura de la red neuronal
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(X, y, epochs=1000, verbose=0)

# Evaluar el rendimiento del modelo
loss, accuracy = model.evaluate(X, y)
print(f'Loss: {loss:.4f}, Accuracy: {accuracy:.2f}')

# Visualización del progreso del entrenamiento
plt.figure(figsize=(10, 5))

# Gráfico de la pérdida
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Gráfico de la precisión
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
