# Batch-Normalization-Study

This repository is to study, summarize what batch normalization is for my machine learning study

* [Awesome blog about batch normalization summary](https://r2rt.com/implementing-batch-normalization-in-tensorflow.html)
* [Youtube tutorial from Deeplearning.AI ](https://www.youtube.com/watch?v=tNIpEZLv_eg)

## Motivations

Prior to train neural networks, we need to rescale input data( = features) so that data distribution follows a standard normal distribution. 

Why rescaling is necessary?. To the point, *we want all features equally contribute to modeling*. If certain features has numerically larger values, those features are likely to learn fast their update than rests. (For example, height vs weight two features obviously have very different numerical range over data points)

So, we can apply same idea to hidden layers units! so that model speeds up learning process

