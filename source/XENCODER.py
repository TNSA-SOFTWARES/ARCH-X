import tensorflow as tf
from transformer import Encoder, Decoder

# Define XEncoder class
class XEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate=0.1):
        super(XEncoder, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate)

    def call(self, inputs, training, mask):
        return self.encoder(inputs, training, mask)
