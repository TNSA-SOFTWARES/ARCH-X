
####This code has been Open-Sourced under MIT Licences but the main proprietary code of the Architecture-X3
#or ARCH-X3 is copyrighted by the TNSA AI the developer can use the open-soure module for developrs on the
# python package index pypi.org TNSA any one with the basic knowledge of C programming and CMD or Powershell
# or any Terminal on the OS can train their own LLM based on the TNSA's ARCH-X3 the code can be downloaded through
# the this command in the PyPi downloader PIP >>pip install ARCHX

#(c)TNSA AI 2023 - 2024 (All Rights Reserved)


"""MIT License

Copyright (c) [2024][TNSA]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""

import tensorflow as tf
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

# Load the Wikipedia dataset
dataset = load_dataset("wikipedia", "20220301.simple")

# Extract the text data from the datasetul
texts = dataset['train']['text']

# Define your own tokenizer
class XTokenizer:
    def __init__(self):
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(lower=True, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')

    def fit_on_texts(self, texts):
        self.tokenizer.fit_on_texts(texts)

    def texts_to_sequences(self, texts):
        return self.tokenizer.texts_to_sequences(texts)

    def pad_sequences(self, sequences, maxlen=None, padding='pre', truncating='pre', value=0):
        return tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=maxlen, padding=padding, truncating=truncating, value=value)

# Initialize your tokenizer
tokenizer = XTokenizer()
tokenizer.fit_on_texts(texts)

# Convert texts to sequences
sequences = tokenizer.texts_to_sequences(texts)

# Pad sequences
max_length = 128
padded_sequences = tokenizer.pad_sequences(sequences, maxlen=max_length)

# Define the model architecture
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        scaled_attention_logits = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(tf.cast(self.depth, tf.float32))  # (batch_size, num_heads, seq_len_q, seq_len_k)
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (batch_size, num_heads, seq_len_q, seq_len_k)
        output = tf.matmul(attention_weights, v)  # (batch_size, num_heads, seq_len_q, depth)
        output = tf.transpose(output, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        output = tf.reshape(output, (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
        output = self.dense(output)  # (batch_size, seq_len_q, d_model)
        return output, attention_weights

class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff):
        super(FeedForward, self).__init__()
        self.dense1 = tf.keras.layers.Dense(dff, activation='relu')
        self.dense2 = tf.keras.layers.Dense(d_model)

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return x

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)
        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)
        return out3, attn_weights_block1, attn_weights_block2

class ARCH_X(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, dropout=0.1):
        super(ARCH_X, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.pos_encoding = self.positional_encoding(max_length, d_model)
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, dropout) for _ in range(num_layers)]
        self.dense = tf.keras.layers.Dense(vocab_size)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, enc_output=None, look_ahead_mask=None, padding_mask=None, training=False):
        seq_len = tf.shape(x)[1]
        attention_weights = {}
        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)
            attention_weights[f'decoder_layer{i+1}_block1'] = block1
            attention_weights[f'decoder_layer{i+1}_block2'] = block2
        output = self.dense(x)  # (batch_size, target_seq_len, vocab_size)
        return output, attention_weights

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
        sines = np.sin(angle_rads[:, 0::2])
        cosines = np.cos(angle_rads[:, 1::2])
        pos_encoding = np.concatenate([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

# Define the model hyperparameters
vocab_size = len(tokenizer.tokenizer.word_index) + 1
d_model = 256  # GPT-2 typically uses 768-dimensional hidden states
num_layers = 12  # GPT-2 has 12 transformer layers
num_heads = 12  # GPT-2 uses 12 attention heads
dff = 1024  # This can remain the same or be adjusted based on your specific needs
dropout_rate = 0.1
batch_size = 8

# Initialize the model
model = ARCH_X(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    vocab_size=vocab_size,
    dropout=dropout_rate
)

# Define the learning rate scheduler
learning_rate_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=2e-4,
    decay_steps=10000,
    decay_rate=0.96,
    staircase=True
)

# Define the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_scheduler, clipnorm=1.0)

# Define the loss function
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))  # Mask to avoid considering padding tokens
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask  # Apply the mask

    return tf.reduce_mean(loss_)


@tf.function
def train_step(inp, tar):
    with tf.GradientTape() as tape:
        predictions, _ = model(inp, enc_output=inp, training=True)  # Pass `training` as a keyword argument
        loss = loss_function(tar, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss

# Prepare the dataset
train_dataset = tf.data.Dataset.from_tensor_slices((padded_sequences, padded_sequences)).shuffle(len(padded_sequences)).batch(batch_size)

# Training loop
epochs = 1
for epoch in range(epochs):
    total_loss = 0

    # Create a progress bar for the epochs
    with tqdm(total=len(train_dataset), desc=f'Epoch {epoch + 1}', unit=' batches') as pbar:
        for (batch, (inp, tar)) in enumerate(train_dataset):
            batch_loss = train_step(inp, tar)
            total_loss += batch_loss

            if batch % 100 == 0:
                print(f'Epoch {epoch + 1} Batch {batch} Loss {batch_loss.numpy():.4f}')

            # Update the progress bar
            pbar.set_postfix({'Loss': total_loss / (batch + 1)})
            pbar.update(1)

    print(f'Epoch {epoch + 1} Loss {total_loss / len(train_dataset):.4f}')

# Save the model
model.save('GPFLLaT-1(ARCH-X3)', save_format='tf')

# Manual download of the model:
# You can manually download the saved model directory from the file system.
""" ARCH-X3 for GPFLLaT-1 used in TNSA AI's NGen 2 """

