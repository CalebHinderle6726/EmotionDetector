import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as kerasLayer

from tensorflow.keras import Sequential
from einops.layers.tensorflow import Rearrange

class MLP(kerasLayer.Layer):
    def __init__(self, dim, hiddenDim, dropout=0.0):
        super(MLP, self).__init__()
        self.net = Sequential([
            kerasLayer.Dense(hiddenDim, activation=keras.activations.gelu), 
            kerasLayer.Dropout(dropout),
            kerasLayer.BatchNormalization(),
            kerasLayer.Dense(dim),
            kerasLayer.Dropout(dropout)
        ])

    def call(self, x, training=True):
        return self.net(x, training=training)

class TransEncoderBlock(kerasLayer.Layer):
    def __init__(self, dim, depth, heads, mlpDim, dropout=0.0):
        super(TransEncoderBlock, self).__init__()
        self.layers = []
        self.normAttn = kerasLayer.LayerNormalization()
        self.normMlp = kerasLayer.LayerNormalization()

        for _ in range(depth):
            self.layers.append([
                (kerasLayer.MultiHeadAttention(num_heads=heads, key_dim=dim//heads)),
                (MLP(dim, mlpDim, dropout=dropout))
            ])

    def call(self, x, training=True):
        for attn, mlp in self.layers:
            x += self.normAttn(attn(x, x, training=training))
            x += self.normMlp(mlp(x, training=training))
        return x

class SinusoidalEmbedding(kerasLayer.Layer):
    def __init__(self, length, dim):
        super(SinusoidalEmbedding, self).__init__()
        self.dim = dim

        # compute sin encodings
        positions = tf.range(length, dtype=tf.float32)[:, tf.newaxis]
        divTerm = tf.exp(tf.range(0, dim, 2, dtype=tf.float32) * -tf.math.log(10000.0) / dim)

        # init encoding matrix
        sinusoidalEncodings = tf.zeros((length, dim), dtype=tf.float32)

        # populate indices
        sinusoidalEncodings = tf.Variable(sinusoidalEncodings, trainable=False)
        sinusoidalEncodings[:, 0::2].assign(tf.sin(positions * divTerm))
        sinusoidalEncodings[:, 1::2].assign(tf.cos(positions * divTerm))

        self.sinusoidalEncodings = tf.convert_to_tensor(sinusoidalEncodings, dtype=tf.float32)

    def call(self, x):
        sequence_len = tf.shape(x)[1]  # getting sequence length
        pos_enc = self.sinusoidalEncodings[:sequence_len, :]
        return x + pos_enc
    
class ConTransformer(kerasLayer.Layer):
    def __init__(self, imageSize, patchSize, dim, depth, heads, mlpDim, dropout=0.0):
        super(ConTransformer, self).__init__()

        self.convA = kerasLayer.Conv2D(32, kernel_size=5, strides=1, padding="same")
        self.bnA = kerasLayer.BatchNormalization()
        
        self.convB = kerasLayer.Conv2D(64, kernel_size=5, strides=1, padding="same")
        self.bnB = kerasLayer.BatchNormalization()

        self.convC = kerasLayer.Conv2D(128, kernel_size=5, strides=1, padding="same")
        self.bnC = kerasLayer.BatchNormalization()

        self.convD = kerasLayer.Conv2D(256, kernel_size=5, strides=1, padding="same")
        self.bnD = kerasLayer.BatchNormalization()

        self.convE = kerasLayer.Conv2D(512, kernel_size=3, strides=1, padding="same")
        self.bnE = kerasLayer.BatchNormalization()

        self.convF = kerasLayer.Conv2D(512, kernel_size=3, strides=1, padding="same")
        self.bnF = kerasLayer.BatchNormalization()

        self.relu = kerasLayer.ReLU()

        patchCount = (imageSize // patchSize) ** 2

        # patch and sinusoidal positional embedding
        self.patch_embedding = Sequential([
            Rearrange('b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1=patchSize, p2=patchSize),
            kerasLayer.Dense(units=dim)
        ], name='patch_embedding')
        self.sinEmbedding = SinusoidalEmbedding(patchCount + 1, dim)

        # transformer encoder block
        self.transformer = TransEncoderBlock(dim, depth, heads, mlpDim, dropout)
        
    def call(self, x, training=True):
        x = self.relu(self.bnA(self.convA(x)))
        x = self.relu(self.bnB(self.convB(x)))
        x = kerasLayer.MaxPooling2D(pool_size=(2,2))(x)

        x = self.relu(self.bnC(self.convC(x)))
        x = self.relu(self.bnD(self.convD(x)))
        x = kerasLayer.MaxPooling2D(pool_size=(2,2))(x)
        
        x = self.relu(self.bnE(self.convE(x)))
        x = kerasLayer.MaxPooling2D(pool_size=(2,2))(x)

        x = self.relu(self.bnF(self.convF(x)))
        x = kerasLayer.MaxPooling2D(pool_size=(2,2))(x)

        # embedding
        x = self.patch_embedding(x)
        x = self.sinEmbedding(x)

        # transformer
        x = self.transformer(x, training = training)

        x = x[:, 0]

        return x