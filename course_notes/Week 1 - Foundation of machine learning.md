# Week 1 - Foundation of machine learning

## 📝 Topic Overview

Machine Learning (ML) is a subfield of computer science and artificial intelligence that focuses on designing systems capable of **learning patterns from data** and **making predictions or decisions without being explicitly programmed**.

ML techniques are widely applied in real-world systems such as recommendation engines, fraud detection, computer vision, natural language processing, and predictive analytics. Their value lies in the ability to **scale to large datasets**, **adapt to complex patterns**, and **generalize beyond observed data**.

---

## 🎯 Key Concepts

- **Machine Learning**: A field of study that enables computers to improve their performance on a task through experience (data) rather than explicit rule-based programming.
- **Task (T)**: The real-world problem to be solved, represented by an **unknown target function** f(x) that maps inputs to outputs.
- **Experience (E)**: The dataset used for learning, consisting of input–output examples derived from the task.
- **Performance Measure (P)**: Quantitative metrics used to evaluate how well a model performs the task (e.g., accuracy, error rate, loss).
- **Hypothesis Space (H)**: The set of all functions that a learning algorithm can choose from. This space is defined by the model type (e.g., linear models, polynomial models, neural networks) and its parameters (weights, bias).
- **Hypothesis Function (h)**: A specific function selected from the hypothesis space that approximates the unknown target function.
- **Loss / Cost / Objective Function**: A mathematical function that measures how far the model’s predictions deviate from the true outputs. It guides learning by defining what it means for a model to perform well.
- **Hyperparameter Tuning**: The process of adjusting hyperparameters (e.g., learning rate, model complexity) to improve model performance based on validation data.
- **Occam’s Razor Principle**: When multiple hypotheses explain the data equally well, the simplest one with the fewest assumptions is preferred, as it is more likely to generalize.

---

## 📖 Study Notes

- Machine learning can be viewed as **programming by optimization**, where a model learns by minimizing an error criterion over data rather than following hard-coded rules.
- The learning objective is to identify a hypothesis ( h ∈ H ) such that:
    
    [ h(x) ≈ f(x) ]
    
    where ( f(x) ) is the unknown target function describing the true relationship in the real world.
    
- Simpler hypothesis functions are often favored because they:
    - Require fewer parameters to optimize
    - Are less prone to overfitting
    - Tend to generalize better to unseen data
- A standard machine learning pipeline consists of:
    1. **Dataset (Experience)** – training examples
    2. **Model (Hypothesis Space)** – the form of functions considered
    3. **Loss Function** – measures prediction error
    4. **Optimization Procedure** – finds parameters that minimize loss
- **Generalization** refers to a model’s ability to perform well on **unseen data**, not just on the training dataset.
- **Overfitting** occurs when a model performs very well on training data but poorly on unseen data, typically due to excessive model complexity.
- **Underfitting** occurs when a model is too simple to capture underlying patterns and performs poorly even on training data.
- Common categories of machine learning include:
    - **Supervised Learning** – learning from labeled data
    - **Unsupervised Learning** – discovering structure in unlabeled data
    - **Reinforcement Learning** – learning through interaction and feedback from an environment

---

## 🔑 Mathematical Formulation

The learning task can be expressed using an unknown target function:

[ y = f(x) ]

Where:
- **x** represents input features (attributes)
- **y** represents the output or label
- **f(x)** is the unknown function governing the true relationship

The dataset (experience) is defined as:

[ D = {(x₁, f(x₁)), (x₂, f(x₂)), …, (xₙ, f(xₙ))} ]

Where:
- **xᵢ**: Input features for the i-th example
- **f(xᵢ)**: True output produced by the target function
- **n**: Number of training examples

Each pair ((xᵢ, f(xᵢ))) represents one observation used to infer an approximation of ( f(x) ).

---

## ✅ Summary & Takeaways

This topic establishes the foundational framework of machine learning by formalizing learning as the process of approximating an unknown target function from data. It highlights the importance of defining the task, selecting an appropriate hypothesis space, and optimizing a loss function to achieve strong generalization.

Key ideas such as Occam’s Razor, overfitting versus underfitting, and generalization form the conceptual backbone of all machine learning algorithms. A solid understanding of these principles is essential for designing reliable models and reasoning about their behavior in real-world applications.
