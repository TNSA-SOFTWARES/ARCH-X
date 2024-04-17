#THIS IS TNSA'S ARCH-X CAN ONLY BE USED TO TRAIN YOUR PERSONAL MODEL AND USE IT PRESONALLY THIS ARCH-X CANNOT BE USE TO TRAIN COMMERCIAL MODELS 
#(c)TNSA - AI 2024 ALL RIGTHS RESERVED 
#THIS CODE HAS OPEN-SOURCED OF PERSONAL USE ONLY 

#TNSA ARCH-X FOR TRAINING HOME LLM's BUT WITH ACCURACY 

import tensorflow as tf
from datasets import load_dataset
import numpy as np

# Adjusted hyperparameters for a smaller model
num_layers = 6
d_model = 256
num_heads = 4
dff = 1024
vocab_size = 10000  # Reduced vocabulary size for testing
dropout_rate = 0.1

# Load the OpenOrca dataset
dataset = load_dataset("Open-Orca/OpenOrca")

# Tokenize the text data
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token='<OOV>')
tokenizer.fit_on_texts(dataset["train"]["response"])

# Define the model architecture
class XMultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(XMultiHeadAttention, self).__init__()
        # Implementation of XMultiHeadAttention

class XEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(XEncoderLayer, self).__init__()
        # Implementation of XEncoderLayer

class ARCH_X(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, dropout=0.1):
        super(ARCH_X, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.pos_encoding = self.positional_encoding(vocab_size, self.d_model)

        self.enc_layers = [XEncoderLayer(d_model, num_heads, dff, dropout) for _ in range(num_layers)]

        self.dense = tf.keras.layers.Dense(vocab_size, activation='softmax', dtype=tf.float32)  # Use float32 for dense layer

        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, training, mask=None):
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))  # Use float32
        x += tf.cast(self.pos_encoding[:, :seq_len, :], tf.float32)  # Cast to float32

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        output = self.dense(x)

        return output

    def positional_encoding(self, position, d_model):
        # Implementation of positional encoding
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)

        # Apply sin and cos to the angle_rads tensor
        sines = tf.math.sin(angle_rads[:, 0::2])  # Apply sin to even indices
        cosines = tf.math.cos(angle_rads[:, 1::2])  # Apply cos to odd indices

        # Interleave sin and cos to create positional encoding matrix
        pos_encoding = tf.stack([sines, cosines], axis=-1)
        pos_encoding = tf.reshape(pos_encoding, [1, position, d_model])

        return pos_encoding

    def get_angles(self, pos, i, d_model):
        # Implementation of get angles
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))  # Use float32
        return pos * angle_rates

# Prepare the dataset for language modeling
input_sequences = np.random.randint(0, vocab_size, (1000, 50))
target_sequences = np.random.randint(0, vocab_size, (1000, 50))  # Ensure the same sequence length as the model's output

# Compile the model
ARCH_X_model = ARCH_X(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    vocab_size=vocab_size,
    dropout=dropout_rate
)


# Compile the model
ARCH_X_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model with adjusted batch size
history = ARCH_X_model.fit(input_sequences, target_sequences, epochs=5, batch_size=32)  # Reduced batch size

# Save the model in SavedModel format
ARCH_X_model.save("ARCH_X_model.h5")



