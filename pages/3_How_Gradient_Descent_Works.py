import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Streamlit app title
st.title("Understanding Gradient Descent")

# Linear Equation Explanation
st.write("### What is the Line Equation and Its Role in Gradient Descent?")
st.write(
    "The equation for a straight line, used in linear regression, is given as:\n"
)
st.latex(r"Y = mX + c")
st.write("Where:")
st.write("- **m** represents the slope of the line. It shows how much Y changes for a unit change in X.")
st.write("- **c** represents the intercept on the y-axis. It is the value of Y when X is 0.")

st.write(
    "In Gradient Descent, the line equation serves as the foundation for understanding the relationship between input features (X) and output predictions (Y). The algorithm iteratively adjusts 'm' (slope) and 'c' (intercept) to minimize the error between predicted and actual values."
)
st.write(
    "The ultimate goal is to minimize the value of 'm', as it directly influences the slope of the cost function, driving the optimization process."
)

# Loss Function Explanation
st.write("### What is the Loss Function and Its Role?")
st.write(
    "The loss function quantifies the error or difference between the predicted and actual values. It is the objective function that Gradient Descent aims to minimize."
)
st.write("For example, in linear regression, the loss function is typically the Mean Squared Error (MSE), calculated as:\n")
st.latex(r"MSE = \frac{1}{n} \sum_{i=1}^{n} (Y_i - \hat{Y}_i)^2")
st.write("Where:")
st.write("- **$n$** is the number of data points.")
st.write("- **$Y_i$** represents the actual values.")
st.write("- **$\hat{Y}_i$** represents the predicted values.")  # Correct inline math

st.write(
    "Gradient Descent uses the gradient (or slope) of the loss function with respect to the parameters to make updates and move towards the minimum error."
)

# Gradient Descent Algorithm Explanation
st.write("### Gradient Descent Algorithm")
st.write(
    "Gradient Descent iteratively calculates the next point using the gradient at the current position, scales it (by a learning rate), and subtracts the obtained value from the current position (makes a step). This process can be summarized as follows:\n"
)
st.latex(r"\theta_{new} = \theta_{current} - \eta \cdot \nabla J(\theta)")
st.write("Where:")
st.write("- **$\\theta$** represents the parameters (weights and biases).")  # Proper inline math
st.write("- **$\eta$** is the learning rate, which scales the gradient and controls the step size.")  # Proper inline math
st.write("- **$\\nabla J(\\theta)$** is the gradient of the cost function with respect to $\theta$.")  # Proper inline math




st.write("#### Key Characteristics of the Learning Rate:")
st.write(
    "- **Small Learning Rate:** Leads to slower convergence, taking more iterations to reach the optimal point.\n"
    "- **Large Learning Rate:** May overshoot the minimum or cause the algorithm to diverge completely."
)

st.write("#### Steps in Gradient Descent:")
st.write("1. **Choose a Starting Point:** Initialize the parameters (weights and biases) at random or zero.")
st.write("2. **Calculate the Gradient:** Compute the slope of the cost function at the current position.")
st.write("3. **Update Parameters:** Take a scaled step in the opposite direction of the gradient to minimize the cost function.")
st.write("4. **Repeat Steps 2 and 3:** Continue until one of the following criteria is met:")
st.write("   - Maximum number of iterations is reached.")
st.write("   - The step size becomes smaller than a predefined tolerance, indicating convergence.")

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# Define the function and its derivative
def f(x):
    return x**2

def f_prime(x):
    return 2 * x

# Gradient descent algorithm
def gradient_descent(start, lr, steps):
    x = start
    path = [x]
    for _ in range(steps):
        x = x - lr * f_prime(x)
        path.append(x)
    return np.array(path)

# Parameters
x_vals = np.linspace(-2, 2, 500)
y_vals = f(x_vals)
start = 1.5  # Starting point for gradient descent
steps = 20

# Small learning rate
lr_small = 0.1
path_small = gradient_descent(start, lr_small, steps)

# Large learning rate
lr_large = 0.9
path_large = gradient_descent(start, lr_large, steps)

# Very large learning rate (causes oscillation)
lr_very_large = 1.0
path_very_large = gradient_descent(start, lr_very_large, steps)

# Create a Plotly figure for small learning rate
fig_small = go.Figure()
fig_small.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="lines", name="f(x) = x²", line=dict(color="blue")))
fig_small.add_trace(go.Scatter(x=path_small, y=f(path_small), mode="markers", marker=dict(size=10, color="red")))
fig_small.add_trace(go.Scatter(x=path_small, y=f(path_small), mode="lines", line=dict(color="orange")))

fig_small.update_layout(
    title="Gradient Descent with Small Learning Rate",
    xaxis_title="x",
    yaxis_title="f(x)",
    template="plotly_white",
    showlegend=False
)

# Create a Plotly figure for large learning rate
fig_large = go.Figure()
fig_large.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="lines", name="f(x) = x²", line=dict(color="blue")))
fig_large.add_trace(go.Scatter(x=path_large, y=f(path_large), mode="markers", marker=dict(size=10, color="red")))
fig_large.add_trace(go.Scatter(x=path_large, y=f(path_large), mode="lines", line=dict(color="orange")))

fig_large.update_layout(
    title="Gradient Descent with Large Learning Rate",
    xaxis_title="x",
    yaxis_title="f(x)",
    template="plotly_white", 
    showlegend=False
)

# Create a Plotly figure for very large learning rate (oscillation)
fig_oscillation = go.Figure()
fig_oscillation.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="lines", name="f(x) = x²", line=dict(color="blue")))
fig_oscillation.add_trace(go.Scatter(x=path_very_large, y=f(path_very_large), mode="markers", marker=dict(size=10, color="red")))
fig_oscillation.add_trace(go.Scatter(x=path_very_large, y=f(path_very_large), mode="lines", line=dict(color="orange")))

fig_oscillation.update_layout(
    title="Gradient Descent with Very Large Learning Rate (Oscillation)",
    xaxis_title="x",
    yaxis_title="f(x)",
    template="plotly_white",
    showlegend=False
)

# Display the plots in Streamlit
st.plotly_chart(fig_small)
st.plotly_chart(fig_large)
st.plotly_chart(fig_oscillation)

# Explanation in Streamlit
st.markdown(
    """
    ### Gradient Descent Oscillation Issue

    When the learning rate is too large, the algorithm overshoots the optimal point, leading to oscillations. 
    This behavior prevents convergence, as the steps taken are too big and fail to stabilize near the minimum.

    #### Solution:
    - Start with a relatively larger learning rate and reduce it as the algorithm approaches the minimum (**learning rate decay**).
    - Ensure the learning rate is appropriately tuned for the problem at hand.

    Oscillation can be visualized in the plot above, where the steps do not converge but instead oscillate around the minimum point.
    """
)

