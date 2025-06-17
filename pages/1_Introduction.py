import streamlit as st
import numpy as np
import plotly.graph_objects as go
import base64
from PIL import Image

# Set up the page configuration
st.set_page_config(
    page_title="Gradient Descent & Visualizer",
    page_icon="📊",
    layout="wide"
)

def set_background(image_path):
    """Sets a background image for the Streamlit app with reduced brightness."""
    try:
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode()

        st.markdown(
            f"""
            <style>
            .stApp {{
                background: linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.5)), 
                            url("data:image/png;base64,{encoded_image}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.error(f"Error loading background image: {e}")

# Title and introduction
st.title("Gradient Descent in Machine Learning 📉")

st.markdown(
    r"""
    **Gradient Descent** is one of the most widely used and powerful optimization algorithms in machine learning and deep learning. It is a cornerstone technique for minimizing the error in predictive models by adjusting the model's parameters iteratively. It ensures that machine learning models, from linear regression to complex neural networks, improve their performance by finding the optimal values for their weights, thereby enhancing prediction accuracy. 
    
    #### <font style="color: #FF5733; font-weight: bold;">Optimization refers to the task of either maximizing or minimizing an objective function.</font>

    In mathematical terminology, optimization refers to the task of minimizing or maximizing an objective function \( f(x) \), parameterized by \( x \). Gradient Descent achieves this by utilizing differentiation to compute the slope (gradient) of the function. By analyzing this slope, it determines the direction and magnitude of steps required to move towards the optimal solution. 
    In machine learning, optimization minimizes the cost function parameterized by the model's parameters. The main objective of gradient descent is to minimize the convex function using iteration of parameter updates. Once optimized, these models become powerful tools for Artificial Intelligence and various computer science applications.

    Gradient Descent enables efficient learning of model parameters by iteratively adjusting them to reduce prediction errors and improve accuracy. This systematic approach ensures convergence to the best-fit solution for a given dataset.
    """,
    unsafe_allow_html=True
)

# Plotly Visualization of Gradient Descent
def gradient_descent_visualization():
    # Define the cost function and its derivative
    def cost_function(x):
        return x**2

    def gradient(x):
        return 2 * x

    # Gradient Descent Algorithm
    x_values = []
    y_values = []
    x = 4  # Initial point
    learning_rate = 0.2
    iterations = 20

    for _ in range(iterations):
        x_values.append(x)
        y_values.append(cost_function(x))
        x = x - learning_rate * gradient(x)

    # Generate the cost function curve
    x_curve = np.linspace(-5, 5, 100)
    y_curve = cost_function(x_curve)

    # Create the plotly figure
    fig = go.Figure()

    # Add the cost function curve
    fig.add_trace(
        go.Scatter(
            x=x_curve, 
            y=y_curve, 
            mode="lines", 
            name="Cost Function",
            line=dict(color="blue")
        )
    )

    # Add the gradient descent steps
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="markers+lines",
            name="Gradient Descent Steps",
            marker=dict(color="red", size=8),
            line=dict(color="red", dash="dot")
        )
    )

    # Update layout
    fig.update_layout(
        title="Gradient Descent Visualization",
        xaxis_title="Parameter (x)",
        yaxis_title="Cost Function Value",
        showlegend=True,
        width=800,
        height=500
    )

    return fig

# Add the visualization to the Streamlit app
st.plotly_chart(gradient_descent_visualization())

st.markdown(
    r"""
    ### What is Gradient Descent or Steepest Descent?
    Gradient descent was initially discovered by *Augustin-Louis Cauchy* in the mid-18th century. It is one of the most commonly used iterative optimization algorithms in machine learning to train machine learning and deep learning models. 
    It helps in finding the local minimum of a function.

    #### Key Concepts:
    - **Local Minimum**: The point where the cost function value is the smallest in the nearby region. Moving in the direction of the negative gradient (away from the gradient) helps us reach this point.
    - **Local Maximum**: The point where the cost function value is the largest in the nearby region. Moving in the direction of the positive gradient leads to this point.

    #### Why Use Gradient Descent?
    Imagine you are on a mountain and want to reach the lowest point (valley). You can only take small steps and don’t have a map. Gradient descent helps you decide the direction and size of each step to efficiently find the valley. Similarly, in machine learning, it finds the optimal parameters for minimizing errors.

    #### Steps of Gradient Descent:
    1. **Calculate the Gradient**: Compute the slope of the cost function at the current point using its derivative. This tells us the direction to move.
    2. **Update Parameters**: Adjust the parameters by moving in the opposite direction of the gradient. The step size is determined by the **Learning Rate ($\lambda$)**.
    3. **Repeat**: Continue updating parameters until the cost function stops decreasing significantly (convergence).

    #### What is Learning Rate?
    - The **Learning Rate ($\lambda$)** determines how big or small the steps are when updating parameters.
    - If $\lambda$ is too large, the algorithm might overshoot the minimum and fail to converge.
    - If $\lambda$ is too small, the algorithm will take too long to converge.

    #### Real-Life Analogy:
    Think of gradient descent as walking downhill on a foggy mountain. You assess the steepness (gradient) of the ground around you and decide which direction to move to get to the bottom (minimum cost function value).

    Gradient Descent can be applied to many types of problems, from linear regression to complex deep learning models.
    """
)




