import streamlit as st
import plotly.graph_objects as go
import math
import numpy as np

def main():
    st.title("Limitations of Gradient Descent")

    st.markdown("""
    Gradient descent is a popular optimization algorithm, but it has several limitations. Here's a detailed overview of its challenges:
    """)

    st.header("1. Local Minima or Saddle Points")
    st.write("""
    - **Problem**: In non-convex optimization problems, the cost function can have multiple valleys (local minima) or flat regions (saddle points). Gradient descent may converge to a local minimum instead of the global minimum.
    - **Impact**: Leads to suboptimal solutions, especially in complex models like deep neural networks.
    - **Example**: Training a neural network with a complex loss landscape might result in finding a suboptimal weight configuration.
    """)

    # Plotting local minima and saddle point
    x = [i / 100 for i in range(-200, 201)]  # Finer resolution for smooth curves
    y_local_minima = [x_i**4 - 4*x_i**2 + 2 for x_i in x]  # Example of a local minima function
    y_saddle_point = [x_i**3 for x_i in x]  # Example of a saddle point function

    # Local minima points
    sqrt_2 = math.sqrt(2)
    local_minima_points_x = [-sqrt_2, sqrt_2]
    local_minima_points_y = [x_i**4 - 4*x_i**2 + 2 for x_i in local_minima_points_x]

    fig = go.Figure()

    # Subplot for local minima
    fig.add_trace(go.Scatter(x=x, y=y_local_minima, mode='lines', name='Local Minima'))
    fig.add_trace(go.Scatter(
        x=local_minima_points_x, y=local_minima_points_y,
        mode='markers',
        marker=dict(size=10, color='orange', symbol='circle'),
        name='Local Minima Points'
    ))

    # Subplot for saddle point
    fig.add_trace(go.Scatter(x=x, y=y_saddle_point, mode='lines', name='Saddle Point'))

    # Highlight the saddle point at x=0
    saddle_x = 0
    saddle_y = saddle_x**3
    fig.add_trace(go.Scatter(
        x=[saddle_x], y=[saddle_y],
        mode='markers',
        marker=dict(size=10, color='red', symbol='circle'),
        name='Saddle Point (x=0, y=0)'
    ))

    fig.update_layout(
        title="Visualization of Local Minima and Saddle Point",
        xaxis_title="x",
        yaxis_title="f(x)",
        legend_title="Function",
        height=500,
        yaxis=dict(range=[-10, 10])  # Zoom out to make local minima more visible
    )

    st.plotly_chart(fig)

    # Replacing the second limitation with the provided code
    st.header("2. Choice of Learning Rate")
    st.write(""" 
    - **Problem**: The learning rate determines the step size for updates:
      - A **high learning rate** can cause overshooting, skipping the minimum, or causing divergence.
      - A **low learning rate** can lead to slow convergence, requiring many iterations.
    - **Impact**: Selecting the wrong learning rate can significantly affect training time and accuracy.
    - **Solution**: Learning rate schedules or adaptive optimizers like Adam can help.
    - **Example**: If the learning rate is 0.1 for a sensitive model, the updates may oscillate around the minimum without settling.
    """)

    # Code for plotting Gradient Descent with different learning rates
    def plot_gd_learning_rate():
        # Define the cost function and its gradient
        def cost_function(x):
            return x**2

        def gradient(x):
            return 2 * x

        # Shared function range for visualization
        x_range = np.linspace(-100, 100, 5000)
        y_range = cost_function(x_range)

        # Gradient Descent Simulation
        iterations = 50  # Increased number of iterations
        low_lr = 0.1  # Low learning rate
        high_lr = 1.2  # High learning rate

        # Initialize starting point
        x_low = [8]  # Start far from the minimum
        x_high = [8]  # Start far from the minimum

        # Simulate GD updates
        for _ in range(iterations - 1):
            x_low.append(x_low[-1] - low_lr * gradient(x_low[-1]))
            x_high.append(x_high[-1] - high_lr * gradient(x_high[-1]))

        # Calculate cost values for the paths
        y_low = [cost_function(x) for x in x_low]
        y_high = [cost_function(x) for x in x_high]

        # Create plot for low learning rate
        fig_low = go.Figure()

        # Plot cost function on the left plot
        fig_low.add_trace(go.Scatter(
            x=x_range, y=y_range, mode='lines', line=dict(color='blue', width=2),
            name="Cost Function"
        ))

        # Plot GD path for low learning rate
        fig_low.add_trace(go.Scatter(
            x=x_low, y=y_low, mode='markers+lines', line=dict(color='orange', width=2),
            marker=dict(size=8, symbol='circle'), name="GD Path (Low LR)"
        ))

        # Adjust layout for low learning rate plot
        fig_low.update_layout(
            title="Gradient Descent Path for Low Learning Rate",
            xaxis_title="x", yaxis_title="f(x)", height=500, width=800, showlegend=True
        )
        fig_low.update_yaxes(range=[0, 80])  # Low learning rate y-axis range
        fig_low.update_xaxes(range=[-10, 10])  # Shared x-axis range

        # Create plot for high learning rate
        fig_high = go.Figure()

        # Plot cost function on the right plot
        fig_high.add_trace(go.Scatter(
            x=x_range, y=y_range, mode='lines', line=dict(color='blue', width=2),
            name="Cost Function"
        ))

        # Plot GD path for high learning rate
        fig_high.add_trace(go.Scatter(
            x=x_high, y=y_high, mode='markers+lines', line=dict(color='red', width=2),
            marker=dict(size=8, symbol='circle'), name="GD Path (High LR)"
        ))

        # Adjust layout for high learning rate plot
        fig_high.update_layout(
            title="Gradient Descent Path for High Learning Rate",
            xaxis_title="x", yaxis_title="f(x)", height=500, width=800, showlegend=True
        )

        # Remove restrictions on y-axis and x-axis for the high learning rate plot
        # Let it be large enough to show divergence without limits
        fig_high.update_yaxes(range=[0, 5000])  # Increase y-range for high learning rate
        fig_high.update_xaxes(range=[-70, 70])  # Expand x-range for large divergence

        return fig_low, fig_high

    # Display the plots
    fig_low, fig_high = plot_gd_learning_rate()

    # Show the low learning rate plot
    st.plotly_chart(fig_low)

    # Show the high learning rate plot
    st.plotly_chart(fig_high)

    st.header("3. Sensitive to Scaling")
    st.write("""
    - **Problem**: Gradient descent assumes that all features contribute equally to the gradient. If features have vastly different scales, the optimization path may zigzag inefficiently.
    - **Impact**: Training becomes slower, and convergence to the minimum can be difficult.
    - **Solution**: Normalize or standardize features so they have similar scales.
    - **Example**: When one feature ranges from 0 to 1 and another from 0 to 10,000, the latter dominates updates.
    """)

    st.header("4. Computationally Expensive")
    st.write("""
    - **Problem**: For large datasets, computing gradients for all data points in every iteration (as in batch gradient descent) is time-intensive.
    - **Impact**: Training on large datasets becomes impractical.
    - **Solution**: Use stochastic gradient descent (SGD) or mini-batch gradient descent to update weights using subsets of data.
    - **Example**: A dataset with 1 million samples will require significant computational resources per iteration.
    """)

    st.header("5. Vanishing or Exploding Gradients")
    st.write("""
    - **Problem**: In deep neural networks, gradients can become very small (vanish) or very large (explode) during backpropagation, especially in the earlier layers:
      - **Vanishing gradients**: Updates become negligible, slowing learning.
      - **Exploding gradients**: Updates become too large, destabilizing training.
    - **Impact**: Hinders effective training of deep networks.
    - **Solution**: Use activation functions like ReLU and techniques like gradient clipping or initialization strategies.
    - **Example**: Using sigmoid activation in deep networks often leads to vanishing gradients due to its small derivative.
    """)

    st.header("6. Convergence Issues")
    st.write("""
    - **Problem**: Gradient descent does not guarantee convergence to the global minimum. The final outcome depends heavily on:
      - The initial parameters.
      - The shape of the cost function.
    - **Impact**: Poor initialization or a highly irregular cost surface can cause convergence to suboptimal solutions or excessive iteration requirements.
    - **Solution**: Random initialization, pretraining, or using techniques like Adam or momentum can improve convergence.
    - **Example**: Poor initialization in training a neural network may lead to slow progress toward the minimum.
    """)

    st.markdown("---")
    st.write("By addressing these limitations through advanced techniques and enhancements, gradient descent can be made more robust and efficient.")

if __name__ == "__main__":
    main()
