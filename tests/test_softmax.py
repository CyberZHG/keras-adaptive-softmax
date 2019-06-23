import os
import tempfile
from unittest import TestCase
import numpy as np
from keras_adaptive_softmax.backend import keras
from keras_adaptive_softmax.backend import backend as K
from keras_adaptive_softmax import AdaptiveEmbedding, AdaptiveSoftmax


class TestSoftmax(TestCase):

    def test_no_projection_no_binding(self):
        input_layer = keras.layers.Input(shape=(None,))
        embed_layer = AdaptiveEmbedding(
            input_dim=3,
            output_dim=16,
            return_embeddings=True,
            return_projections=True,
        )(input_layer)
        softmax_layer = AdaptiveSoftmax(input_dim=16, output_dim=3)(embed_layer)
        model = keras.models.Model(input_layer, softmax_layer)
        model_path = os.path.join(tempfile.gettempdir(), 'test_ada_embed_%f.h5' % np.random.random())
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects={
            'AdaptiveEmbedding': AdaptiveEmbedding,
            'AdaptiveSoftmax': AdaptiveSoftmax,
        })
        model.summary()

    def test_no_projection_bind(self):
        input_layer = keras.layers.Input(shape=(None,))
        embed_layer = AdaptiveEmbedding(
            input_dim=3,
            output_dim=16,
            return_embeddings=True,
            return_projections=True,
        )(input_layer)
        softmax_layer = AdaptiveSoftmax(
            input_dim=16,
            output_dim=3,
            bind_embeddings=True,
            bind_projections=True,
        )(embed_layer)
        model = keras.models.Model(input_layer, softmax_layer)
        model_path = os.path.join(tempfile.gettempdir(), 'test_ada_embed_%f.h5' % np.random.random())
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects={
            'AdaptiveEmbedding': AdaptiveEmbedding,
            'AdaptiveSoftmax': AdaptiveSoftmax,
        })
        model.summary()

    def test_single_projection_no_binding(self):
        input_layer = keras.layers.Input(shape=(None,))
        embed_layer = AdaptiveEmbedding(
            input_dim=3,
            output_dim=16,
            embed_dim=5,
            return_embeddings=True,
            return_projections=True,
        )(input_layer)
        softmax_layer = AdaptiveSoftmax(input_dim=16, output_dim=3, embed_dim=5)(embed_layer)
        model = keras.models.Model(input_layer, softmax_layer)
        model_path = os.path.join(tempfile.gettempdir(), 'test_ada_embed_%f.h5' % np.random.random())
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects={
            'AdaptiveEmbedding': AdaptiveEmbedding,
            'AdaptiveSoftmax': AdaptiveSoftmax,
        })
        model.summary()

    def test_single_projection_bind(self):
        input_layer = keras.layers.Input(shape=(None,))
        embed_layer = AdaptiveEmbedding(
            input_dim=3,
            output_dim=16,
            embed_dim=5,
            return_embeddings=True,
            return_projections=True,
        )(input_layer)
        softmax_layer = AdaptiveSoftmax(
            input_dim=16,
            output_dim=3,
            embed_dim=5,
            bind_embeddings=True,
            bind_projections=True,
        )(embed_layer)
        model = keras.models.Model(input_layer, softmax_layer)
        model_path = os.path.join(tempfile.gettempdir(), 'test_ada_embed_%f.h5' % np.random.random())
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects={
            'AdaptiveEmbedding': AdaptiveEmbedding,
            'AdaptiveSoftmax': AdaptiveSoftmax,
        })
        model.summary()

    def test_cutoffs_no_binding(self):
        input_layer = keras.layers.Input(shape=(None,))
        embed_layer = AdaptiveEmbedding(
            input_dim=30,
            output_dim=3,
            embed_dim=8,
            cutoffs=[10, 20, 25],
            div_val=2,
            mask_zero=True,
            return_embeddings=True,
            return_projections=True,
        )(input_layer)
        softmax_layer = AdaptiveSoftmax(
            input_dim=3,
            output_dim=30,
            embed_dim=8,
            cutoffs=[10, 20, 25],
            div_val=2,
        )(embed_layer)
        model = keras.models.Model(input_layer, softmax_layer)
        model_path = os.path.join(tempfile.gettempdir(), 'test_ada_embed_%f.h5' % np.random.random())
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects={
            'AdaptiveEmbedding': AdaptiveEmbedding,
            'AdaptiveSoftmax': AdaptiveSoftmax,
        })
        model.summary()

    def test_cutoffs_bind(self):
        input_layer = keras.layers.Input(shape=(None,))
        embed_layer = AdaptiveEmbedding(
            input_dim=30,
            output_dim=3,
            embed_dim=8,
            cutoffs=[10, 20, 25],
            div_val=2,
            mask_zero=True,
            return_embeddings=True,
            return_projections=True,
        )(input_layer)
        softmax_layer = AdaptiveSoftmax(
            input_dim=3,
            output_dim=30,
            embed_dim=8,
            cutoffs=[10, 20, 25],
            div_val=2,
            bind_embeddings=True,
            bind_projections=True,
        )(embed_layer)
        model = keras.models.Model(input_layer, softmax_layer)
        model_path = os.path.join(tempfile.gettempdir(), 'test_ada_embed_%f.h5' % np.random.random())
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects={
            'AdaptiveEmbedding': AdaptiveEmbedding,
            'AdaptiveSoftmax': AdaptiveSoftmax,
        })
        model.summary()
