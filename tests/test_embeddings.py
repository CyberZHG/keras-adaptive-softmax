from unittest import TestCase
import numpy as np
from keras_adaptive_softmax.backend import keras
from keras_adaptive_softmax.backend import backend as K
from keras_adaptive_softmax import AdaptiveEmbedding


class TestEmbedding(TestCase):

    def test_sample_default(self):
        input_layer = keras.layers.Input(shape=(None,))
        embed_layer = AdaptiveEmbedding(input_dim=3, output_dim=5, return_embeddings=True)(input_layer)
        func = K.function([input_layer], embed_layer)
        outputs = func([np.array([[0, 1, 2]])])
        self.assertTrue(np.allclose(outputs[0], outputs[1]))
