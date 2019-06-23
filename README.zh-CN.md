# Keras Adaptive Softmax

[![Travis](https://travis-ci.org/CyberZHG/keras-adaptive-softmax.svg)](https://travis-ci.org/CyberZHG/keras-adaptive-softmax)
[![Coverage](https://coveralls.io/repos/github/CyberZHG/keras-adaptive-softmax/badge.svg?branch=master)](https://coveralls.io/github/CyberZHG/keras-adaptive-softmax)
[![Version](https://img.shields.io/pypi/v/keras-adaptive-softmax.svg)](https://pypi.org/project/keras-adaptive-softmax/)
![Downloads](https://img.shields.io/pypi/dm/keras-adaptive-softmax.svg)
![License](https://img.shields.io/pypi/l/keras-adaptive-softmax.svg)

![](https://img.shields.io/badge/keras-tensorflow-blue.svg)
![](https://img.shields.io/badge/keras-theano-blue.svg)
![](https://img.shields.io/badge/keras-tf.keras-blue.svg)
![](https://img.shields.io/badge/keras-tf.keras/eager-blue.svg)
![](https://img.shields.io/badge/keras-tf.keras/2.0_beta-blue.svg)

\[[中文](https://github.com/CyberZHG/keras-adaptive-softmax/blob/master/README.zh-CN.md)|[English](https://github.com/CyberZHG/keras-adaptive-softmax/blob/master/README.md)\]

## 安装

```bash
pip install keras-adaptive-softmax
```

## 使用


`AdaptiveEmbedding`和`AdaptiveSoftmax`一般同时使用。`AdaptiveEmbedding`进行不定长的嵌入，`AdaptiveSoftmax`根据网络输出计算其与嵌入的相似度：

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

`cutoffs`和`div_val`控制着不同词嵌入的长度，`cutoffs`是嵌入长度开始缩减的下标，`div_val`是每次长度缩减的倍数。在上面的例子中：

* 一共30个词
* 前5个词嵌入长度为32
* 接下来15 - 5 = 10个词嵌入长度为16
* 再接下来25 - 15 = 10个词嵌入长度为8
* 最后30 - 25 = 5个词嵌入长度为4

一般情况下，`AdaptiveEmbedding`中`return_embeddings`和`return_projections`都要设置为`True`，这一层返回值中第一个是嵌入进行线性映射到同一尺寸的结果，后续是嵌入和映射的权重，用于`AdaptiveSoftmax`中相似度的计算。`AdaptiveSoftmax`中`bind_embeddings`和`bind_projections`都可以接受一个列表，列表的长度与不同长度嵌入的个数相同，控制每一种嵌入长度是否使用`AdaptiveEmbedding`中已有的权重，如果对应位置设为`False`则会生成新的可训练权重。
