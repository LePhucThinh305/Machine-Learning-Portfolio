# Week 2 - Regression

## 📝 Topic Overview

Regression is a foundational supervised learning technique used to predict continuous numerical values based on input features. This summary captures the core theoretical understanding of regression models, the optimization process via gradient descent, and important practical considerations for effective implementation.

## 🎯 Key Concepts

- Gradient descent is a way to optimize the min value of the loss function.
- Hyperparameter: A parameter that is part of the machine learning algorithm not the model or hypothesis. An example of this would be the step size to optimize loss function.
- All the attributes must be independent so that we can take the partial derivative of the weight in gradient descent for multivariate regression.

## 📖 Study Notes

- There are 2 types of supervised learning: regression and classification.
- Regression:
    - Experience:
        
        The experience of regression is a data set of inputs and outputs.
        
        Attributes of the task x:
        
        - real-value, continuous.
        - independent variable.
        - x have one attribute are called univariate.
        - x have many attributes are called multivariate.
        
        Outputs: y, real value, continuous, dependent variable.
        
    - Hypothesis space of regression:
        - Can be linear or non-linear.
        - Linear formula in formula section
        
        The goal is to find the best regression line that best fit the model such that the total distance (error) between the regression line and actual training set (x,y) is minimal.
        
    - Lost function:
        - This function measure the performance of the regression line by calculating the average error of every x ( h(x) - y) ^ 2.
        - The best regression will minimize the lost function (min J(θ0, θ1)).
        - We can use slope of the lost function where θ0 = 0 to find θ1 that give min J(θ0, θ1).
- In univariate hypothesis there are only 2 parameter of the hypothesis function:  θ0 and θ1.
- In multivariate hypothesis there are m + 1 parameters where m is the total number of attributes.

- Gradient descent is a technique to optimize the lost function using the value of the gradient of the graph (theta 1, loss function). Since in univariate the loss function is a hyperbola. We can take the partial derivative and increase theta 0/1 (if gradient is negative) or decrease (if gradient is positive). The value we increase/decrease by is (learning rate × derivative).
- In multivariate the gradient descent is a multi dimensional function since for m attributes there are m+1 weight. So to update the values of thetas we take partial derivative of each theta with respect to the loss function and adjust the values of each theta by adding/removing (learning rate × derivative) just like in univariate.

- Practical issues:
    - Hypothesis:
        - Is the regression model (choice of weights) good? That is, the hypothesis h(x) is a good choice from hypothesis space h.
        - Is the linear regression suitable? That is does h(x) roughly equal to f(x)?
    - Prediction:
        - We can test if the model is good by testing on unseen datasets.
    - Practical matters:
        - Are the properties of the datasets independent because if it is not we cannot take the partial derivative.
        - Limited time.
        - Limited computation power.
    - Feature scaling:
        - The domains of different features can be very different. If one domain have much bigger range the hypothesis function will be affected more by it. So we want to scale all the features to the same proportion.
        - Standardization or normalization formula can be applied to attributes so that we center data around 0 and remove bias caused by large baseline values.

- Polynomial regression:
    - Linear regression is too simple and does not have enough capacity to represent data.
    
    ![image.png](image.png)
    
    - Assume that the target function (learning task) is a polynomial equation. Thus u can add power to the attributes x1 to xm.
    - We do this by changing the features polynomial features and fit linear regression to new features.
    - For example in univariate with x as attributes. We can add another attributes x^2 to turn it into a multivariate.

## 📊 Visualizations & Graphs

![image.png](image%201.png)

- Output of regression is continuous value (e.g. price of stock)
- Output of classification is non continuous value (e.g. cat/dog, red/blue)

## 🔑 Important Formulas

- Formula for linear regression:

$$
\hat{y} = w_1 x_1 + w_2 x_2 + \cdots + w_n x_n + b
$$

Where x are the attributes of the task

W is the weight of each attributes

ŷ is the hypothesis with respect to weight W

- Loss function:

$$
J(\theta_0, \theta_1) = \frac{1}{2n} \sum_{i=1}^{n} \left( (\theta_1 x_i + \theta_0) - y_i \right)^2
$$

Where θ0 and θ1 are the coefficient of the loss function.

Y is the actual output of the training data set.

- Standardization:

$$
x_j(i) = ( x_j(i) - μ_j ) / σ_j
$$

## ✅ Summary & Takeaways

This study covers the fundamental
 theory of regression analysis in machine learning, tracing the journey 
from basic concepts to advanced practical applications.

### **Core Concepts Mastered:**

1. **Regression Fundamentals:** Understood as a supervised learning task where the goal is to map input features (independent variables) to a continuous output (dependent
variable) using a hypothesis function (e.g., a linear model).
2. **Model Optimization:** Learned that the "best" model is found by minimizing a **Loss Function** (Mean Squared Error) that quantifies the error between predictions and actual values. This is achieved through the **Gradient Descent** algorithm, which iteratively adjusts model parameters (weights) by moving against the gradient of the loss function.
3. **From Simple to Complex Models:**
    - Started with **Univariate Linear Regression** (one feature, two parameters).
    - Expanded to **Multivariate Linear Regression** (many features, m+1 parameters).
    - Progressed to **Polynomial Regression** by engineering new features from existing ones, allowing the fitting of non-linear relationships while still using linear regression
    techniques.
4. **Critical Practical Implementation Skills:**
    - **Feature Scaling (Standardization/Normalization):** Essential for ensuring gradient descent converges efficiently by
    treating all features equally, regardless of their original scale.
    - **Assumption Awareness:** Recognized the importance of feature independence for gradient descent
    and the need to evaluate if a linear model is suitable for the given
    task.
    - **Evaluation Mindset:** Differentiated between optimizing the hypothesis (finding best weights) and evaluating the suitability of the hypothesis space itself (is
    linear regression the right choice?).
