import tensorflow as tf
import numpy as np

# Define a simple dataset (example)
# Input features
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
# Target outputs (logical OR)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)

# Define a neural network model using Keras API
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(2,))
])

# Compile the model
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model
history = model.fit(X, y, epochs=1000, verbose=0)

# Evaluate the model
loss, accuracy = model.evaluate(X, y)
print(f'Loss: {loss:.4f}, Accuracy: {accuracy:.2f}')
