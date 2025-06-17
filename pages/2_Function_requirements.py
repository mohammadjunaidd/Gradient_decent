import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy.optimize import fsolve

st.set_page_config(page_title="Function Requirements")

st.title("Function Requirements for Gradient Descent")

# Set the markdown explanation
st.markdown(
    """
    To effectively use gradient descent, a function must satisfy two key conditions:  

    1. **Differentiability** - The function must have a derivative (or gradient) at every point.  
    2. **Convexity** - The function should ideally be convex, ensuring a single global minimum.  

    ## 1. Differentiability

    To make gradient descent work effectively, the function must be continuous at every point of its domain. 
    This means there should be no sudden jumps, breaks, or discontinuities. Additionally, the function should be differentiable, 
    meaning that at every point, we should be able to compute a derivative (or gradient), which tells us how the function is changing.

    Differentiability ensures that the gradient descent algorithm can work smoothly by using the gradient to guide us towards the minimum value.

    ### Why does differentiability matter for gradient descent?
    
    Gradient descent relies on the gradient (derivative) of a function to iteratively adjust its parameters. 
    If the function is not differentiable, there will be no well-defined gradient at certain points, 
    which can cause gradient descent to fail or behave unpredictably. This is crucial when you're trying to find the minimum value of the function, 
    as the gradient guides the search direction.

    **Key Concepts:**
    - **Continuity**: A function is continuous if there are no breaks or jumps in its graph. 
    - **Differentiability**: A function is differentiable if it has a well-defined derivative at every point in its domain.

    ### Example of Differentiable Function

    Consider the simple quadratic function:

    $$
    f(x) = x^2
    $$

    The function $f(x) = x^2$ is continuous and differentiable. Let's compute its derivative step-by-step:

    1. The function is $f(x) = x^2$.
    2. Apply the power rule of differentiation: 
    $$ 
    \\frac{d}{dx} (x^n) = n \\cdot x^{n-1}
    $$ 
    For $f(x) = x^2$, we get:
    $$ 
    \\frac{d}{dx}(x^2) = 2x
    $$ 
    3. The derivative of $f(x) = x^2$ is $f'(x) = 2x$, which tells us the slope of the curve at any point $x$.

    ### Visualization of $f(x) = x^2$ and its derivative:

    """
)

# Plot f(x) = x^2
x = np.linspace(-10, 10, 500)
y1 = x**2  # f(x) = x^2

# Create the first plotly figure for f(x) = x^2
fig1 = go.Figure()

# Plot f(x) = x^2
fig1.add_trace(
    go.Scatter(x=x, y=y1, mode='lines', name='f(x) = x²', line=dict(color='blue'))
)

# Update layout for f(x) = x^2
fig1.update_layout(
    title="Plot of f(x) = x²",
    xaxis_title="x",
    yaxis_title="y",
    showlegend=True
)

# Display the first plot in Streamlit
st.plotly_chart(fig1)

# Plot f'(x) = 2x
y2 = 2 * x  # f'(x) = 2x

# Create the second plotly figure for f'(x) = 2x
fig2 = go.Figure()

# Plot f'(x) = 2x
fig2.add_trace(
    go.Scatter(x=x, y=y2, mode='lines', name="f'(x) = 2x", line=dict(color='red'))
)

# Update layout for f'(x) = 2x
fig2.update_layout(
    title="Plot of f'(x) = 2x",
    xaxis_title="x",
    yaxis_title="y",
    showlegend=True
)

# Display the second plot in Streamlit
st.plotly_chart(fig2)

# Add more explanation
st.markdown(
    """
    ### Example of Non-Differentiable Function

    Now, consider the absolute value function:

    $$ 
    f(x) = |x|
    $$

    1. For $x > 0$, $f(x) = x$, so the derivative is $f'(x) = 1$.
    2. For $x < 0$, $f(x) = -x$, so the derivative is $f'(x) = -1$.
    3. However, at $x = 0$, the function has a sharp corner, and the derivative is undefined.

    ### Visualization of the Absolute Value Function:

    """

)

# Plot f(x) = |x| (absolute value function)
y3 = np.abs(x)  # f(x) = |x|

# Create the third plotly figure for f(x) = |x|
fig3 = go.Figure()

# Plot f(x) = |x|
fig3.add_trace(
    go.Scatter(x=x, y=y3, mode='lines', name="f(x) = |x|", line=dict(color='green'))
)

# Update layout for f(x) = |x|
fig3.update_layout(
    title="Plot of f(x) = |x|",
    xaxis_title="x",
    yaxis_title="y",
    showlegend=True
)

# Display the third plot in Streamlit
st.plotly_chart(fig3)

# Add more explanation for non-differentiable function
st.markdown(
    """
    The sharp corner at $x = 0$ makes this function non-differentiable at that point. 
    The gradient descent algorithm would not work well at this point, as there's no clear gradient to guide the optimization process.

    ### Gradient

    #### <font style="color: #FF5733; font-weight: bold;">Differentiation in higher dimensions is equivalent to the concept of a **gradient**.</font>
    
    A gradient represents the slope of a curve at a specific point in a given direction. For a univariate function, it is the first derivative. For a multivariate function, it is a vector of partial derivatives:

    $$
    \\nabla f(p) = \\left[ \\frac{\\partial f}{\\partial x_1}, \\frac{\\partial f}{\\partial x_2}, \\dots, \\frac{\\partial f}{\\partial x_n} \\right]
    $$

    Here, ∇ (nabla) represents the gradient operator. The gradient points in the direction of the steepest ascent of the function, and the magnitude of the gradient tells you how steep that ascent is.

    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    - For **scalar-valued functions**, differentiation is represented as **d/dx**, which gives the rate of change of the function with respect to x.
    - For **vector-valued functions**, the gradient is denoted as **grad(f)**, and it consists of partial derivatives for each variable.

    ### Conclusion

    For gradient descent to work efficiently, the function must be continuous and differentiable. 
    Differentiability ensures that we can compute a gradient at each point, which is crucial for guiding the algorithm towards the minimum. 
    Non-differentiable points, like sharp corners, can cause problems for gradient-based optimization methods.
    """
)


st.markdown("""
    ## 2. Convexity

    The second requirement is that the function must be convex. For a univariate function, a function is convex if the line segment connecting any two points on the curve lies entirely on or above the curve. If the line crosses below the curve, it indicates the presence of a local minimum that is not a global minimum.
    
    **Mathematically, A function** *f(x)* **is convex if, for any two points** *x<sub>1</sub>* **and** *x<sub>2</sub>* **in its domain, and for any value** *λ* **such that 0 ≤ λ ≤ 1, the following inequality holds:**
""", unsafe_allow_html = True)

st.latex(r"""
    f(\lambda x_1 + (1 - \lambda) x_2) \leq \lambda f(x_1) + (1 - \lambda) f(x_2)
""")

st.markdown(
    """
    ### Example of convex and non-convex functions:
    """
)
# Plot Convex Function f(x) = x^2 and its derivative
x = np.linspace(-3, 3, 500)
y_convex = x**2  # Convex function: f(x) = x^2
y_prime_convex = 2 * x  # Derivative: f'(x) = 2x

# Create the plotly figure for convex function
fig_convex = go.Figure()

# Plot convex function f(x) = x^2
fig_convex.add_trace(
    go.Scatter(x=x, y=y_convex, mode='lines', name='f(x) = x²', line=dict(color='blue'))
)

# Plot derivative f'(x) = 2x
fig_convex.add_trace(
    go.Scatter(x=x, y=y_prime_convex, mode='lines', name="f'(x) = 2x", line=dict(color='orange'))
)

# Update layout for convex function
fig_convex.update_layout(
    title="Convex Function f(x) = x² and Its Derivative",
    xaxis_title="x",
    yaxis_title="y",
    showlegend=True
)

# Display the convex function plot
st.plotly_chart(fig_convex)

# Plot Non-Convex Function f(x) = sin(x) + 0.1x^2
x_non_convex = np.linspace(-10, 10, 500)
y_non_convex = np.sin(x_non_convex) + 0.1 * x_non_convex**2  # Non-convex function
y_prime_non_convex = np.cos(x_non_convex) + 0.2 * x_non_convex  # Derivative

# Create the plotly figure for non-convex function
fig_non_convex = go.Figure()

# Plot the non-convex function
fig_non_convex.add_trace(
    go.Scatter(x=x_non_convex, y=y_non_convex, mode='lines', name='f(x) = sin(x) + 0.1x²', line=dict(color='blue'))
)

# Plot the derivative of the non-convex function
fig_non_convex.add_trace(
    go.Scatter(x=x_non_convex, y=y_prime_non_convex, mode='lines', name="f'(x) = cos(x) + 0.2x", line=dict(color='red'))
)

# Update layout for non-convex function
fig_non_convex.update_layout(
    title="Non-Convex Function and Its Derivative",
    xaxis_title="x",
    yaxis_title="y",
    showlegend=True
)

# Display the non-convex function plot
st.plotly_chart(fig_non_convex)

# Explanation of convex vs non-convex functions
st.markdown(
    """
    ### Semi-Convex Functions and Saddle Points

    A **semi-convex function** combines convexity in some regions with non-convexity elsewhere. A **saddle point** is a critical point where the gradient is zero, but it’s neither a local minimum nor maximum. It features:

    - Downward curvature in some directions.
    - Upward curvature in others.

    Saddle points can slow or stall Gradient Descent due to vanishing gradients, especially in high-dimensional spaces.

    - <span style="color: red;">&#x25CF;</span> **Saddle Point**: A point where the gradient is zero but is not a local minimum or maximum.
    - <span style="color: yellow;">&#x25CF;</span> **Critical Point**: A point where the gradient of the function equals zero. Critical points can be local minima, local maxima, or saddle points.
    - **Critical Points** can also be referred to as stationary points.
    """,
    unsafe_allow_html=True
)



# Define the x values and semi-convex function
x = np.linspace(-3, 3, 500)
y_semi_convex = x**4 - 4*x**2 + x  # Semi-convex function with a saddle point: f(x) = x^4 - 4x^2 + x

# Create the plotly figure for semi-convex function
fig_semi_convex = go.Figure()

# Plot semi-convex function f(x) = x^4 - 4x² + x
fig_semi_convex.add_trace(
    go.Scatter(x=x, y=y_semi_convex, mode='lines', name='f(x) = x⁴ - 4x² + x', line=dict(color='blue'))
)

# Function representing the derivative
def derivative(x):
    return 4*x**3 - 8*x + 1

# Solve for the critical points where f'(x) = 0
critical_points = fsolve(derivative, [-2, 0, 2])

# Get the values of the function at the critical points
critical_values = np.interp(critical_points, x, y_semi_convex)

# Add the saddle points to the plot
fig_semi_convex.add_trace(
    go.Scatter(x=critical_points, y=critical_values, mode='markers', name='Critical Points', marker=dict(color='yellow', size=10))
)

# Highlight the origin as a saddle point
fig_semi_convex.add_trace(
    go.Scatter(x=[0], y=[0], mode='markers+text', name='Saddle Point', 
               marker=dict(color='red', size=12), textposition="top center")
)

# Calculate y-values for the yellow line endpoints
y_yellow_start = (-2)**4 - 4*(-2)**2 + (-2)
y_yellow_end = (3)**4 - 4*(3)**2 + (3)

# Add yellow line from x = -2 to x = 3
fig_semi_convex.add_trace(
    go.Scatter(x=[-2, 3], y=[y_yellow_start, y_yellow_end], mode='lines', line=dict(color='purple', width=2), showlegend=False)
)

# Add markers for the start and end of the yellow line
fig_semi_convex.add_trace(
    go.Scatter(x=[-2, 3], y=[y_yellow_start, y_yellow_end], mode='markers', 
               marker=dict(color='purple', size=10), showlegend=False)
)

# Calculate y-values for the purple line endpoints
y_purple_start = (-2.8)**4 - 4*(-2.8)**2 + (-2.8)
y_purple_end = (2.2)**4 - 4*(2.2)**2 + (2.2)

# Add purple line from x = -2.8 to x = 2.2
fig_semi_convex.add_trace(
    go.Scatter(x=[-2.8, 2.2], y=[y_purple_start, y_purple_end], mode='lines', line=dict(color='orange', width=2), showlegend=False)
)

# Add markers for the start and end of the purple line
fig_semi_convex.add_trace(
    go.Scatter(x=[-2.8, 2.2], y=[y_purple_start, y_purple_end], mode='markers', 
               marker=dict(color='orange', size=10), showlegend=False)
)

# Update layout for semi-convex function f(x) = x⁴ - 4x² + x
fig_semi_convex.update_layout(
    title="Plot of Semi-Convex Function f(x) = x⁴ - 4x² + x (with Saddle Points and Annotations)",
    xaxis_title="x",
    yaxis_title="y",
    showlegend=True
)

# Display the semi-convex function plot
st.plotly_chart(fig_semi_convex)