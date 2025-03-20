import tensorflow as tf #type: ignore

class PerceptronTF(tf.keras.Model):
    def __init__(self, input_dim, output_dim, activation='relu'):
        super(PerceptronTF, self).__init__()
        self.dense = tf.keras.layers.Dense(
            units=output_dim,
            activation=activation,
            input_shape=(input_dim,)
        )
    
    def call(self, inputs):
        return self.dense(inputs)