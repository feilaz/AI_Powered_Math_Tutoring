import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import os
import threading
from langchain.tools import StructuredTool
from langchain_core.tools import tool
from typing import Optional, Union, List
from pydantic import BaseModel, Field
from sympy import solve, Symbol, parse_expr
from sympy.parsing.sympy_parser import standard_transformations, implicit_multiplication, convert_xor

matplotlib.use('Agg')

@tool("draw_function_graph")
def draw_function_graph(func_str: str, xmin: float = -10, xmax: float = 10, num_points: int = 400) -> str:
    """
    Draws a graph for a mathematical function.

    Parameters:
      func_str (str): A string representing a function of x (e.g. "np.sin(x)", "x**2", "math.log(x)").
      xmin (float): Minimum x value.
      xmax (float): Maximum x value.
      num_points (int): Number of points in the graph.
      show_popup (bool): If True and running on the main thread, display the graph in a popup window;
                         otherwise, the graph is saved as an image and opened with the default OS viewer.
                         
    Returns:
      A string message indicating success.
    """
    x = np.linspace(xmin, xmax, num_points)
    
    # Use a safe dictionary for eval, including math and numpy functions.
    safe_dict = {"x": x, "np": np, "math": math}
    try:
        y = eval(func_str, {"__builtins__": {}}, safe_dict)
    except Exception as e:
        return f"Error evaluating function: {e}"

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, label=f"f(x) = {func_str}")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title(f"f(x) = {func_str}")
    plt.legend()
    plt.grid(True)
    
    # Check if we're on the main thread and pop-ups are allowed.
    # Switch to a non-interactive backend for safety.
    matplotlib.use('Agg')
    output_file = "function_graph.png"
    plt.savefig(output_file)
    plt.close()
    # Open the image using the default Windows image viewer (non-blocking).
    os.startfile(output_file)
    msg = f"Graph saved as {output_file} and opened."
    
    plt.close()
    return msg




class EquationInput(BaseModel):
    equation: str = Field(description="Mathematical equation to solve (e.g., '2*x + 3 = 7')")
    variable: Optional[str] = Field(default='x', description="Variable to solve for")

@tool("solve_equation", args_schema=EquationInput)
def solve_equation(equation: str, variable: Optional[str] = None) -> Union[List[str], str]:
    """
    Solves a given algebraic equation using SymPy.

    Args:
        equation (str): The equation to solve (e.g., "x^2 - 4 = 0").
        variable (Optional[str]): The variable to solve for (e.g., "x").

    Returns:
        Union[List[str], str]: A list of solutions as strings or an error message.
    """
    # Split the equation into LHS and RHS
    if '=' not in equation:
        return "Invalid equation: No equals sign found."
    lhs_str, rhs_str = equation.split('=', 1)
    
    # Configure transformations for parsing (handles ^ and implicit multiplication)
    transformations = standard_transformations + (implicit_multiplication, convert_xor)
    
    try:
        # Parse both sides of the equation
        lhs = parse_expr(lhs_str.strip(), transformations=transformations)
        rhs = parse_expr(rhs_str.strip(), transformations=transformations)
    except Exception as e:
        return f"Error parsing equation: {e}"
    
    # Formulate the equation as LHS - RHS = 0
    equation = lhs - rhs
    
    # Identify symbols in the equation
    symbols = equation.free_symbols
    
    # Handle constant equations (no variables)
    if not symbols:
        return "All real numbers are solutions." if equation == 0 else "No solution."
    
    # Determine the variable to solve for
    if variable:
        var_sym = Symbol(variable)
        if var_sym not in symbols:
            return f"Variable '{variable}' not found in the equation."
    else:
        if len(symbols) > 1:
            return "Multiple variables detected. Please specify the variable to solve for."
        var_sym = symbols.pop()
    
    # Solve the equation
    try:
        solutions = solve(equation, var_sym)
    except NotImplementedError:
        return "Unable to solve the equation with SymPy."
    
    # Handle results
    if not solutions:
        return "No solution found."
    return [str(sol) for sol in solutions]