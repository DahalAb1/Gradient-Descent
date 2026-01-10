# Understanding Gradient Descent

This repository documents my attempt to understand gradient descent by implementing it from scratch, rather than relying on built-in machine learning libraries.

## What I Built

A Jupyter notebook that implements linear regression without using any external libraries for the machine learning method. The goal was to understand the underlying mathematics by writing the algorithm myself.

## Mathematical Foundation

### Loss Function (Mean Squared Error)

$$f = \frac{1}{m} \sum_{i=1}^{m} (y_{\text{pred}} - y_{\text{actual}})^2$$

where $m$ is the number of samples.

### Linear Prediction Model

$$y_{\text{pred}} = \theta_1 \cdot x + \theta_0$$

where $\theta_1$ is the weight (slope) and $\theta_0$ is the bias (y-intercept).

### Combined Loss Function

$$f(\theta_1, \theta_0) = \frac{1}{m} \sum_{i=1}^{m} (\theta_1 x_i + \theta_0 - y_i)^2$$

### Gradient Computation

Partial derivative with respect to weight:

$$\frac{\partial f}{\partial \theta_1} = \frac{2}{m} \sum_{i=1}^{m} \text{error}_i \cdot x_i$$

Partial derivative with respect to bias:

$$\frac{\partial f}{\partial \theta_0} = \frac{2}{m} \sum_{i=1}^{m} \text{error}_i$$


### Parameter Update Rule

$$\theta_1 := \theta_1 - \alpha \cdot \frac{\partial f}{\partial \theta_1}$$

$$\theta_0 := \theta_0 - \alpha \cdot \frac{\partial f}{\partial \theta_0}$$


## What I Learned

I always understood mathematics behind gradient descent, but I had never implemented it's working into program. This exercise helped me build a strong intuition about how equations transpires into working code.  

## Files

- `gradient_descent.ipynb` - Complete implementation with code, visualizations, and mathematical explanations
