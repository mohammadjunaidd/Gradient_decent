import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sympy import symbols, diff, lambdify, sin, cos, tan, exp, log
import sympy as sp

def gradient_descent(formula, start, learning_rate, iterations=50, threshold=1e10, grad_clip_value=1e5):
    x_sym = symbols('x')
    try:
        formula_sym = eval(formula, {"x": x_sym, "sin": sin, "cos": cos, "tan": tan, "exp": exp, "log": log, "np": np})
        derivative = diff(formula_sym, x_sym)
        grad_func = lambdify(x_sym, derivative, modules="numpy")
    except Exception as e:
        st.error(f"Error in formula evaluation: {e}")
        return []

    x = start
    trajectory = [x]

    for _ in range(iterations):
        try:
            grad = grad_func(x)
            grad = np.clip(grad, -grad_clip_value, grad_clip_value)
            x = x - learning_rate * grad
            if abs(x) > threshold:
                st.error("Gradient descent diverged due to large values.")
                break
            trajectory.append(x)
        except Exception as e:
            st.error(f"Error in gradient calculation: {e}")
            break

    return trajectory

st.set_page_config(layout="wide")

st.title("Gradient Descent Visualizer")

if "iterations" not in st.session_state:
    st.session_state.iterations = 0
if "formula" not in st.session_state:
    st.session_state.formula = "x**2"
if "start_point" not in st.session_state:
    st.session_state.start_point = 5.0
if "learning_rate" not in st.session_state:
    st.session_state.learning_rate = 0.25
if "show_all_iterations" not in st.session_state:
    st.session_state.show_all_iterations = False
if "tolerance" not in st.session_state:
    st.session_state.tolerance = 0.01

def get_ml_formula(algorithm):
    formulas = {
        "Linear Regression": "(x - 2)**2",
        "Logistic Regression": "log(1 + exp(-x))"
    }
    return formulas.get(algorithm, "x**2")

with st.sidebar:
    st.write("## Inputs")
    formula_input = st.text_input("Function", value=st.session_state.formula)
    if formula_input != st.session_state.formula:
        st.session_state.formula = formula_input
        st.session_state.iterations = 0
        st.session_state.trajectory = []  # Reset trajectory

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("x^2"):
            st.session_state.formula = "x**2"
            st.session_state.iterations = 0
            st.session_state.trajectory = []  # Reset trajectory
    with col2:
        if st.button("sin(x)"):
            st.session_state.formula = "sin(x)"
            st.session_state.iterations = 0
            st.session_state.trajectory = []  # Reset trajectory
    with col3:
        if st.button("cos(x)"):
            st.session_state.formula = "cos(x)"
            st.session_state.iterations = 0
            st.session_state.trajectory = []  # Reset trajectory

    col4, col5, col6 = st.columns(3)
    with col4:
        if st.button("tan(x)"):
            st.session_state.formula = "tan(x)"
            st.session_state.iterations = 0
            st.session_state.trajectory = []  # Reset trajectory
    with col5:
        if st.button("exp(x)"):
            st.session_state.formula = "exp(x)"
            st.session_state.iterations = 0
            st.session_state.trajectory = []  # Reset trajectory
    with col6:
        if st.button("log(x)"):
            st.session_state.formula = "log(x)"
            st.session_state.iterations = 0
            st.session_state.trajectory = []  # Reset trajectory

    st.write("## ML Optimized Equations")
    col7, col8 = st.columns(2)
    with col7:
        if st.button("Linear Regression"):
            st.session_state.formula = get_ml_formula("Linear Regression")
            st.session_state.iterations = 0
            st.session_state.trajectory = []  # Reset trajectory
    with col8:
        if st.button("Logistic Regression"):
            st.session_state.formula = get_ml_formula("Logistic Regression")
            st.session_state.iterations = 0
            st.session_state.trajectory = []  # Reset trajectory

    start_point_input = st.number_input("Starting Point", value=st.session_state.start_point)
    if start_point_input != st.session_state.start_point:
        st.session_state.start_point = start_point_input
        st.session_state.iterations = 0
        st.session_state.trajectory = []  # Reset trajectory

    learning_rate_input = st.number_input("Learning Rate", value=st.session_state.learning_rate, min_value=0.01, step=0.01)
    if learning_rate_input != st.session_state.learning_rate:
        st.session_state.learning_rate = learning_rate_input
        st.session_state.iterations = 0
        st.session_state.trajectory = []  # Reset trajectory

    tolerance_input = st.number_input("Tolerance", value=st.session_state.tolerance, min_value=0.001, step=0.001)
    if tolerance_input != st.session_state.tolerance:
        st.session_state.tolerance = tolerance_input
        st.session_state.iterations = 0
        st.session_state.trajectory = []  # Reset trajectory

    next_iteration_button = st.button("Next Iteration", key="next_iteration_button", use_container_width=True)
    if next_iteration_button:
        st.session_state.iterations += 1

    toggle_iterations_button = st.button("Show All Iterations", key="toggle_iterations_button", use_container_width=True)
    if toggle_iterations_button:
        st.session_state.show_all_iterations = not st.session_state.show_all_iterations

    zoom_factor = st.slider("Zoom Level", min_value=1, max_value=30, value=10, step=1)

trajectory = []
try:
    x_sym = symbols('x')
    formula = st.session_state.formula
    formula = formula.replace('np.maximum', 'Max').replace('np.piecewise', 'Piecewise')
    
    # Ensure formula contains only valid functions and replace Max correctly
    formula = formula.replace('Max', 'np.maximum')

    formula_sym = eval(formula, {"x": x_sym, "sin": sin, "cos": cos, "tan": tan, "exp": exp, "log": log, "np": np})
    
    y_func = lambdify(x_sym, formula_sym, 'numpy')

    # Handling log(x) specifically to avoid errors for x <= 0
    def safe_y_func(x):
        if st.session_state.formula == "log(x)":
            x = np.clip(x, 1e-10, None)  # Avoid log(0) or negative values
        return y_func(x)

    trajectory = gradient_descent(st.session_state.formula, st.session_state.start_point, st.session_state.learning_rate, iterations=st.session_state.iterations)
except Exception as e:
    st.error(f"Error in formula evaluation: {e}")

if trajectory:
    current_x = trajectory[-1]
    previous_x = trajectory[-2] if len(trajectory) > 1 else None
    difference = abs(current_x - previous_x) if previous_x is not None else None
    if difference is not None and difference <= st.session_state.tolerance:
        st.success(f"Found minima at position: {current_x:.4f}")

x_start = min(trajectory) - zoom_factor if trajectory else -10
x_end = max(trajectory) + zoom_factor if trajectory else 10
x = np.linspace(x_start, x_end, 500)
try:
    y = safe_y_func(x)
except Exception as e:
    st.error(f"Error in generating graph: {e}")
    y = None

if y is not None:
    fig = go.Figure()

    # Add the main function line
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=f"y = {st.session_state.formula}"))

    # Add trajectory points
    if trajectory:
        fig.add_trace(go.Scatter(x=trajectory[:-1], y=[safe_y_func(t) for t in trajectory[:-1]],
                                 mode='markers', name="Previous Points", marker=dict(color='yellow')))
        fig.add_trace(go.Scatter(x=[trajectory[-1]], y=[safe_y_func(trajectory[-1])],
                                 mode='markers', name="Current Point", marker=dict(color='red', size=10)))

        # Add tangent line
        grad_at_current = (safe_y_func(current_x + 1e-5) - safe_y_func(current_x)) / 1e-5
        tangent_y = grad_at_current * (x - current_x) + safe_y_func(current_x)
        fig.add_trace(go.Scatter(x=x, y=tangent_y, mode='lines', name="Tangent Line", line=dict(color='orange')))

    # Customize layout
    fig.update_layout(title=f"Graph of {st.session_state.formula}",
                      xaxis_title="x", yaxis_title="y",
                      template="plotly_white",
                      width=1000, height=600)

    st.plotly_chart(fig)

if trajectory:
    current_x = trajectory[-1]
    previous_x = trajectory[-2] if len(trajectory) > 1 else None
    difference = current_x - previous_x if previous_x is not None else None
    current_data = {
        "Iteration": [st.session_state.iterations],
        "Current X": [f"{current_x:.2f}"],
        "Previous X": [f"{previous_x:.2f}" if previous_x is not None else "N/A"],
        "Difference": [f"{difference:.2f}" if difference is not None else "N/A"]
    }
    st.table(current_data)

if trajectory and st.session_state.show_all_iterations:
    all_iterations_data = []
    for i, val in enumerate(trajectory):
        if i == 0:
            continue
        prev_val = trajectory[i - 1]
        diff_val = val - prev_val
        all_iterations_data.append({
            "Iteration": i,
            "Current X": f"{val:.2f}",
            "Previous X": f"{prev_val:.2f}",
            "Difference": f"{diff_val:.2f}"
        })
    st.table(all_iterations_data)
