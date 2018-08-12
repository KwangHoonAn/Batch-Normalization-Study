# Batch-Normalization-Study

This repository is to study, summarize what batch normalization is for my machine learning study

* [Awesome blog about batch normalization summary](https://r2rt.com/implementing-batch-normalization-in-tensorflow.html)
* [Youtube tutorial from Deeplearning.AI ](https://www.youtube.com/watch?v=tNIpEZLv_eg)

## Motivations

Prior to train neural networks, we need to rescale input data( = features) so that data distribution follows a standard normal distribution. 

Why rescaling is necessary?. To the point, **we want all features equally contribute to modeling**. 
If certain features has numerically larger values, those features are likely to learn fast their update than rests. (For example, height vs weight two features obviously have very different numerical range over data points)

So, we can apply same idea to hidden layers units! 
**To be shorts, batch normalization is to rescaling of ouputs of the hidden units at hidden layers!**

**Note : according to the literacture on Youtube, applying batch normalization over input of the activation function is better than normalizing over output of neurons**

### Problem in neural networks

Let's think about cat detection scenario, Neural networks model has been trained using "Black cat" with a label "cat" (Example from the youtube tutorial). If "Orange cat" is tested, performance would not be excellent as we want. This is because the data distribution of black cat, **P(black cat)** and **P(Orange cat)** are slightly different and Our training objective has been trained with **P(cat|black cat)**. This problem is called **Internal Covariate Shift**

In perspective of neural network learning porcess, input from earlier layers will keep chainging during training by different input data distribution ( Internal Covariate shift ). This will make next layers to adopt earlier layers distribution, which means **next layers will suffer from Internal Covariate shift**.

### Mathematical term explanation
Other parts are very tedious, calculate mean and variance of each batch.
Gamma, beta are trainable parameters these two parameters help layers distribution to have non-zero mean(centering in non-zero), epsilon is needed to avoid zero denominator case

### Regularization effect

Inputs within mini-batch will be scaled by batch normalization. In other words, mini batch inputs will no longer keep **whole data distribution** by being scaled by **mini batch mean and variance**. We can consider as **noise is added to the inputs**. This will play a role similar to dropout