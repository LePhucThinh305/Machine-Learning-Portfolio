# Week 3 - Logistic Regression and Regularization

## 📝 Topic Overview

This module introduces **logistic regression** as a core supervised learning algorithm for **binary classification** problems. It explains how linear model outputs are transformed into probabilities using the **sigmoid function**, and how **decision thresholds** are used to make class predictions. The module also covers **loss functions for classification** and introduces **regularization** as a method to reduce overfitting and improve model generalization.

## 🎯 Key Concepts

- Classification is a type of supervised learning that have discrete labels/classes for output.
- A threshold/cut off is a point we set to convert **linear hypothesis function** with output that have values out of 0 to 1 range to 0 and 1.
- A decision point is the point at which binary classification change (0 or 1), it is applied to a probability produced by **logistic regression function**.
- Sigmoid function also known as the logistic function have output between 0 and 1 and also continuous and smooth so we can take derivative. It is used for binary classification hypothesis function because linear regression model does not fit the output data well.
- Logistic regression is just putting the output of linear regression into the input of a sigmoid function so that we convert all the output in the range of 0 to 1.
- Regularization is a process to discourage the complexity of the function model to prevent overfitting. This is achieved by adding a penalty (regularization term) into the loss function to make the weights and bias smaller.
- Lambda is a hyperparameter to adjust the regularization weight and is part of the regularization term. Big values of lambda will make thetas much smaller and small value of lambda will reduce theta much less.

## 📖 Study Notes

- We can divide the output of classifications into 2 classes:
    - Binary class: 2 output ( 1 or 2, used or not used, benign/malignant).
    - Tertiary class (n-array class):  more than 2 output. For example many animal label (cat, dog, tiger, fish). However computer only understand number so we can define cat as 1, dog as 2, tiger as 3. The meaning of each number we will define before we train the model.
- In binary classification the output is only 0 or 1. However a function a linear function h(x) can produce values that are more than 1 and less than 0. To convert them to 0 or 1 we set a threshold, a cut off to divide the classes. For example, for a model that determine pass or fail we can set threshold at 50% if its over or equal it is a pass, lower than cut off it is a fail.
- The value of the threshold is one of the hyper-parameter that we should fine tune.
- Interpretation of the Hypothesis Function:
    - In univariate classification using **logistic regression**, the hypothesis function is defined as:
    
    **hθ(x) = g(θ0 + θ1x)**
    
    - where the sigmoid function **g(z)** is:
    
    **g(z) = 1 / (1 + e⁻ᶻ)**
    
    - The hypothesis function represents the **estimated probability** that the output label is 1 given an input **x**:
    
    **hθ(x) = P(y = 1 | x; θ)**
    
    - Because the sigmoid function maps any real-valued input into the range **(0, 1)**, the output of **hθ(x)** can be directly interpreted as a probability.
    - Example:
    
    If **hθ(x) = 0.8**, the model estimates an **80% probability** that **y = 1** for the given input **x**.
    
- Since the sum of probability of y = 1 given x and theta and y= 0 given x and theta = 1 since we can infer the formula to calculate probability of one knowing the other
- When a decision point is chosen we can deduce the output of linear regression function that give that exact decision point in the logistic regression function. Then we can classify output of the linear hypothesis regression without putting it through the sigmoid function.

- **Regularization:**
    
    Suppose we want to train the model for a dataset with **stars** as **training datasets** and **blue dot** as **unseen data point**:
    
    - We can see that the second model is better fit than the first one based on the loss values (less error between model and actual value)
    - The third model loss function is smaller than second model because all the training data fall on the model function.
    - **However the second model predict the unseen data more accurately, less loss value compared to model 3. This is called overfitting.**
    - So if you can use loss function for training and testing. If you minimize loss value too much your model can run into overfitting and fail to accurately predict training data.
    - Underfitting is when the model cannot predict the training dataset and testing dataset.

![image.png](image.png)

To prevent the overfitting problem without changing the shape of the hypothesis. We can minimize the theta coefficients by reducing it toward zero, discouraging more complex polynomial. This process is called **regularization**.

To do this we introduced a **regularization term** into the loss function:

$$
J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (y^{(i)} - \theta^T x^{(i)})^2 + \lambda \sum_{j=1}^{n} \theta_j^2
$$

- Lambda is the weight of the regularization the bigger it is the more regularization thus making thetas lower, function less complex, more linear and vice versa. Do note that this mean if lambda is too big it will make the function underfit and if lambda is too small the function will overfit.

- In multi-classification the way we compute a model is to split it into many binary classifications to check one versus all. For example a model that have 4 classes: A, B, C, D will have 4 binary classification to check A or not A, B or not B, C or not C, D or not D.

## 🔑 Important Formulas

- Logistic regression function for binary classification:

$$
h_θ(x) = g(θᵀx) = 1 / (1 + e^(−θᵀx))
$$

*We use the value of the regression function and place it into the sigmoid function input (z).

- Probability of y = 0 given x and theta

$$
P(y = 0 \mid x; \theta) = 1 - h_\theta(x)
$$

This work because h(x) = P(y=1 ∣ x; θ)

- Regularization of the loss function (Ridge regression):

$$
J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (y^{(i)} - \theta^T x^{(i)})^2 + \lambda \sum_{j=1}^{n} \theta_j^2
$$

## ✅ Summary & Takeaways

Logistic regression models the probability of class membership rather than predicting continuous values. The sigmoid function enables probabilistic interpretation, while decision thresholds convert probabilities into class labels. Regularization penalizes overly complex models, helping prevent overfitting and improving performance on unseen data. Together, these concepts form a foundational understanding of modern classification models used in practical machine learning systems.
