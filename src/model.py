import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.saving import register_keras_serializable

# 🔹 Register Patchify so Keras can find it when saving/loading
@register_keras_serializable(package="Custom")
class Patchify(layers.Layer):
    def __init__(self, patch_size=8, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, inputs):
        """
        Splits the time dimension into patches of fixed size
        """
        # inputs shape: [batch, time, features]
        shape = tf.shape(inputs)
        batch, time, features = shape[0], shape[1], shape[2]

        # Ensure divisibility
        num_patches = time // self.patch_size
        inputs = inputs[:, :num_patches * self.patch_size, :]

        patches = tf.reshape(inputs, (batch, num_patches, self.patch_size * features))
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config
        
    @classmethod
    def from_config(cls, config):
        return cls(**config)


@register_keras_serializable(package="Custom")
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length=500, output_dim=256, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.output_dim = output_dim
        self.position_embedding = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim
        )

    def call(self, inputs):
        # Create position indices using Keras backend ops
        positions = tf.range(start=0, limit=tf.shape(inputs)[1], delta=1)
        embedded_positions = self.position_embedding(positions)
        return inputs + embedded_positions

    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "output_dim": self.output_dim,
        })
        return config
        
    @classmethod
    def from_config(cls, config):
        return cls(**config)

def build_transformer_audio_model(num_classes: int, input_shape=(None, 64)):
    """
    A simple Transformer-based classifier for audio spectrograms.
    """
    inputs = keras.Input(shape=input_shape, name="log_mel_input")

    # Apply patchify
    x = Patchify()(inputs)
    
    # Dense projection to match embedding dimension
    x = layers.Dense(256)(x)
    
    # Add positional embedding
    x = PositionalEmbedding()(x)

    # First Transformer block
    attention_output = layers.MultiHeadAttention(
        num_heads=4, key_dim=64
    )(x, x)
    x = layers.Dropout(0.1)(attention_output)
    x = layers.Add()([x, attention_output])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Feed-forward network
    ffn = layers.Dense(1024, activation="relu")(x)
    ffn = layers.Dense(256)(ffn)
    ffn = layers.Dropout(0.1)(ffn)
    x = layers.Add()([x, ffn])
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    # Pooling + output layers
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="AudioTransformer")
