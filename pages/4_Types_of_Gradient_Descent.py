import streamlit as st
import pandas as pd
from PIL import Image

# Title and Introduction
st.title("Diving Deeper into Gradient Descent and its Variants")
st.write("""
Gradient Descent (GD), Stochastic Gradient Descent (SGD), and Mini-Batch Gradient Descent are fundamental optimization algorithms in machine learning, particularly for training neural networks. 
This page provides an in-depth explanation of these concepts and their nuances.
""")

# Add Image
image_path = "D:\Github\Publication\gradient\pages\Typesofgd.jpg"
image = Image.open(image_path)
st.image(image, caption="Gradient Descent and its Variants", use_container_width=True)

# Explanation: Gradient Descent (GD)
st.subheader("1. Gradient Descent (GD)")
st.write("""
**Concept**:
- Gradient Descent (GD) is a first-order optimization algorithm that iteratively updates model parameters to minimize the error or loss of a function.
- In GD, the entire dataset is used to compute the gradient of the loss function at each iteration, ensuring a globally informed update step.
- It is widely used in machine learning due to its ability to find optimal solutions by systematically moving toward the function's minima.

**Key Characteristics**:
- Works best for convex loss functions where a global minimum exists.
- Progresses smoothly towards convergence but is computationally expensive for large datasets.

**Process**:
1. **Compute the Gradient**: The gradient is calculated for the entire training dataset to find the direction of the steepest ascent in the loss function.
2. **Update Parameters**: Each parameter is adjusted in the opposite direction of the gradient, scaled by the learning rate (α).
""")

# Mathematical Representation with LaTeX for GD
st.write("**Mathematical Representation**:")
st.latex(r"""
\theta = \theta - \alpha \cdot \nabla J(\theta)
""")
st.write("""
- **θ**: The current parameter values  
- **α**: The learning rate  
- **∇J(θ)**: The gradient of the loss function J with respect to θ
""")

# Explanation: Stochastic Gradient Descent (SGD)
st.subheader("2. Stochastic Gradient Descent (SGD)")
st.write("""
**Concept**:
- Stochastic Gradient Descent (SGD) is an iterative optimization algorithm that updates model parameters based on a single randomly selected data point.
- Unlike GD, which processes the entire dataset at each step, SGD performs updates more frequently, leading to faster but noisier parameter adjustments.
- This randomness introduces variability in the path toward convergence, making SGD suitable for large-scale datasets and dynamic environments.

**Key Characteristics**:
- Often preferred for online learning, where data arrives sequentially.
- While the updates can oscillate, this noise can help escape local minima and saddle points in non-convex problems, such as deep learning models.

**Process**:
1. **Select a Random Example**: A single training example is randomly chosen from the dataset.
2. **Compute the Gradient**: The gradient is calculated based on this one example, which approximates the overall direction of the steepest descent.
3. **Update Parameters**: Parameters are updated using the calculated gradient and learning rate.
""")

# Mathematical Representation with LaTeX for SGD
st.write("**Mathematical Representation**:")
st.latex(r"""
\theta = \theta - \alpha \cdot \nabla J(\theta_i)
""")
st.write("""
- **θ**: The current parameter values  
- **α**: The learning rate  
- **∇J(θᵢ)**: The gradient of the loss function J with respect to θ for the i-th example
""")

# Explanation: Mini-Batch Gradient Descent (MBGD)
st.subheader("3. Mini-Batch Gradient Descent (MBGD)")
st.write("""
**Concept**:
- Mini-Batch Gradient Descent (MBGD) combines the best aspects of GD and SGD by dividing the dataset into small batches (mini-batches) and performing updates on these subsets.
- By averaging the gradients over the mini-batch, MBGD strikes a balance between the computational efficiency of SGD and the stability of GD.
- Mini-batches are typically chosen to fit the available memory, making MBGD highly scalable and practical for modern machine learning frameworks.

**Key Characteristics**:
- Reduces variance in updates compared to SGD, leading to more stable convergence.
- Can leverage parallel processing and GPU acceleration for faster gradient computations, making it the go-to method for training deep neural networks.

**Process**:
1. **Select a Mini-Batch**: A small random subset of the training dataset (e.g., 32, 64, or 128 examples) is chosen.
2. **Compute the Gradient**: The gradient is averaged over the mini-batch, reducing noise compared to single-example updates in SGD.
3. **Update Parameters**: Parameters are updated based on the averaged gradient and learning rate.
""")

# Mathematical Representation with LaTeX for MBGD
st.write("**Mathematical Representation**:")
st.latex(r"""
\theta = \theta - \alpha \cdot \frac{1}{m} \sum_{i=1}^m \nabla J(\theta_i)
""")
st.write("""
- **θ**: The current parameter values  
- **α**: The learning rate  
- **m**: The mini-batch size  
- **∇J(θᵢ)**: The gradient of the loss function J with respect to θ for the i-th example in the mini-batch
""")

# Key Considerations
st.subheader("Key Considerations")
st.write("""
1. **Learning rate (α)**: A crucial hyperparameter that determines the step size in each iteration. A well-chosen learning rate is essential for efficient convergence.
2. **Batch size**: The number of training examples in a mini-batch. Larger batch sizes can lead to more stable updates but may require more memory.
3. **Convergence**: All three methods aim to minimize the loss function, but the convergence behavior can differ. GD tends to converge smoothly, while SGD and mini-batch GD can exhibit fluctuations.
""")

# Conclusion
st.write("""
By understanding the nuances of these gradient descent methods, practitioners can effectively train machine learning models, particularly deep neural networks, on diverse datasets.
""")

# About Iteration
st.subheader("What is an Iteration?")
st.write("""
An **iteration** refers to a single update of the model's parameters during the training process.  
Each iteration involves calculating the gradient of the loss function and using it to adjust the parameters.  
The number of iterations required for convergence depends on the optimization method, learning rate, and the complexity of the problem.
""")

# Comparison Table with Proper Styling
st.subheader("Comparison of Gradient Descent Types")
st.write("""
The table below summarizes the key differences between Gradient Descent (GD), Stochastic Gradient Descent (SGD), and Mini-Batch Gradient Descent (MBGD):
""")

data = {
    "Aspect": [
        "Dataset Usage per Iteration",
        "Update Frequency",
        "Computational Efficiency",
        "Convergence Behavior",
        "Noise in Updates",
        "Memory Requirement"
    ],
    "Gradient Descent (GD)": [
        "Entire dataset",
        "Low (one update per epoch)",
        "High (requires processing the full dataset)",
        "Smooth convergence",
        "Minimal",
        "High (depends on dataset size)"
    ],
    "Stochastic Gradient Descent (SGD)": [
        "One randomly selected example",
        "High (one update per example)",
        "Low (processes one example at a time)",
        "Fluctuates due to noise",
        "High",
        "Low (requires storing only one example)"
    ],
    "Mini-Batch Gradient Descent (MBGD)": [
        "A subset of the dataset",
        "Moderate (one update per mini-batch)",
        "Moderate (batch size controls trade-off)",
        "Relatively smooth",
        "Moderate",
        "Moderate (depends on mini-batch size)"
    ],
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Display the table
st.write("### Comparison of Gradient Descent Types")
st.table(df)

