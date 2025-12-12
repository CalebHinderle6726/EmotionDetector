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
            kerasLayer.Dense(dim),
            kerasLayer.Dropout(dropout)
        ])

    def call(self, x, training=True):
        return self.net(x, training=training)

class TransEncoderBlock(kerasLayer.Layer):
    def __init__(self, dim, depth, heads, mlpDim, dropout=0.0):
        super(TransEncoderBlock, self).__init__()
        self.layers = []

        for _ in range(depth):
            self.layers.append({
                "norm_attn": kerasLayer.LayerNormalization(),
                "attn": kerasLayer.MultiHeadAttention(num_heads=heads, key_dim=dim // heads),
                "norm_mlp": kerasLayer.LayerNormalization(),
                "mlp": MLP(dim, mlpDim, dropout=dropout),
            })

    def call(self, x, training=True):
        for layer in self.layers:
            x += layer["norm_attn"](layer["attn"](x, x, training=training))
            x += layer["norm_mlp"](layer["mlp"](x, training=training))
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
        sequenceLen = tf.shape(x)[1]  # getting sequence length
        
        # allowing for mixed precision
        posEnc = tf.cast(self.sinusoidalEncodings[:sequenceLen, :], x.dtype)
        return x + posEnc
    
class ConTransformer(kerasLayer.Layer):
    def __init__(self, imageSize, patchSize, dim, depth, heads, mlpDim, dropout=0.0):
        super(ConTransformer, self).__init__()

        self.downsampleFactor = 2 ** 3
        if imageSize % (patchSize * self.downsampleFactor) != 0:
            raise ValueError(
                f"imageSize must be divisible by patchSize {self.downsampleFactor}."
            )
        patchGrid = imageSize // (patchSize * self.downsampleFactor)
        patchCount = patchGrid ** 2

        self.convA = kerasLayer.Conv2D(64, kernel_size=5, strides=1, padding="same")
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
        self.pool = kerasLayer.MaxPooling2D(pool_size=(2, 2))

        # patch and sinusoidal positional embedding
        self.patchEmbedding = Sequential([
            Rearrange('b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1=patchSize, p2=patchSize),
            kerasLayer.Dense(units=dim)
        ], name='patch_embedding')
        self.sinEmbedding = SinusoidalEmbedding(patchCount, dim)

        # transformer encoder block
        self.transformer = TransEncoderBlock(dim, depth, heads, mlpDim, dropout)
        
    def call(self, x, training=True):
        x = self.relu(self.bnA(self.convA(x)))
        x = self.relu(self.bnB(self.convB(x)))
        x = self.pool(x)

        x = self.relu(self.bnC(self.convC(x)))
        x = self.relu(self.bnD(self.convD(x)))
        x = self.pool(x)
        
        x = self.relu(self.bnE(self.convE(x)))
        x = self.pool(x)

        x = self.relu(self.bnF(self.convF(x)))

        # embedding
        x = self.patchEmbedding(x)
        x = self.sinEmbedding(x)

        # transformer
        x = self.transformer(x, training = training)

        x = tf.reduce_mean(x, axis=1)

        return x