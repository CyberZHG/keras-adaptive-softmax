# Keras Adaptive Softmax

[![Travis](https://travis-ci.org/CyberZHG/keras-adaptive-softmax.svg)](https://travis-ci.org/CyberZHG/keras-adaptive-softmax)
[![Coverage](https://coveralls.io/repos/github/CyberZHG/keras-adaptive-softmax/badge.svg?branch=master)](https://coveralls.io/github/CyberZHG/keras-adaptive-softmax)
[![Version](https://img.shields.io/pypi/v/keras-adaptive-softmax.svg)](https://pypi.org/project/keras-adaptive-softmax/)
![Downloads](https://img.shields.io/pypi/dm/keras-adaptive-softmax.svg)
![License](https://img.shields.io/pypi/l/keras-adaptive-softmax.svg)

![](https://img.shields.io/badge/keras-tensorflow-blue.svg)
![](https://img.shields.io/badge/keras-tf.keras-blue.svg)

\[[中文](https://github.com/CyberZHG/keras-adaptive-softmax/blob/master/README.zh-CN.md)|[English](https://github.com/CyberZHG/keras-adaptive-softmax/blob/master/README.md)\]

## Install

```bash
pip install keras-adaptive-softmax
```

## Usage

Generally, `AdaptiveEmbedding` and `AdaptiveSoftmax` should be used together. `AdaptiveEmbedding` provides variable length embeddings, while `AdaptiveSoftmax` calculates the similarities between the outputs and the generated embeddings.

```python
import keras
from keras_adaptive_softmax import AdaptiveEmbedding, AdaptiveSoftmax

input_layer = keras.layers.Input(shape=(None,))
embed_layer = AdaptiveEmbedding(
    input_dim=30,
    output_dim=32,
    cutoffs=[5, 15, 25],
    div_val=2,
    return_embeddings=True,
    return_projections=True,
    mask_zero=True,
)(input_layer)
dense_layer = keras.layers.Dense(
    units=32,
    activation='tanh',
)(embed_layer[0])
softmax_layer = AdaptiveSoftmax(
    input_dim=32,
    output_dim=30,
    cutoffs=[5, 15, 25],
    div_val=2,
    bind_embeddings=True,
    bind_projections=True,
)([dense_layer] + embed_layer[1:])
model = keras.models.Model(inputs=input_layer, outputs=softmax_layer)
model.compile('adam', 'sparse_categorical_crossentropy')
model.summary()
```

`cutoffs` and `div_val` controls the length of embeddings for each token. Suppose we have 30 distinct tokens, in the above example:

* The lengths of the embeddings of the first 5 tokens are 32
* The lengths of the embeddings of the next 10 tokens are 16
* The lengths of the embeddings of the next 10 tokens are 8
* The lengths of the embeddings of the last 5 tokens are 4
