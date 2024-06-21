import streamlit as st
from random import randint
from threading import Thread
from time import sleep
import matplotlib.pyplot as plt
from numpy import linspace
from PIL import Image
from AzizKitten import sqrt, cbrt, sin, cos, tan, cot, sec, csc, asin, acos, atan, acot, asec, acsc, sinh, cosh, tanh, sech, coth, floor, ceil, gcd, lcm, factorial, integrate, log, exp, ln, derivative, inf, e, pi

def plot_sqrt(x):
    return x**.5

def plot_cbrt(x):
    return abs(x)/x*abs(x)**(1/3)

def plot_ln(x):
    return (x**1e-10 - 1)/1e-10

def plot_asin(x):
    def integrand(t):
        return 1/sqrt(1-t**2)
    return integrate(integrand, 0, x)

def equation_solver(expression: str, real: bool=True, cplx: bool=False, max_solutions: int=None , interval_start: float=float("-inf"), interval_end: float=float("inf"), deprived_start: float=None, deprived_end: float=None, deprived_values: list=None) -> list:
    """
    return the solutions for the giving expression.
    # Exemples:
    - equation_solver("abs(sin(x)) = 0.5", interval_start=0, interval_end=2*pi, deprived_values=[5*pi/6])
    
 [0.5235987756, 3.6651914292, 5.7595865316]
    # Important:
    Allowed variable is only x.
    # Why I don't get any result?
    - Check the expression.
    - Check the used functions.
    - Give another try (Bot sometimes find it hard to solve from the first run).
    """
    if type(max_solutions) is int:
        if max_solutions < 0:
            raise ValueError("Amount of solutions cannot be a negative number.")
    elif max_solutions == None:
        pass
    else:
        raise TypeError("Value input must be a positive integer.")
    if not expression.count("=") == 1:
        raise SyntaxError("The expression must contain only one '='.")
    left_side = (expression.split("=")[0].replace(" ", "")).replace("^","**")
    right_side = (expression.split("=")[1].replace(" ","")).replace("^","**")
    if len(left_side) == left_side.count(" ") or len(right_side) == right_side.count(" "):
        raise SyntaxError("Left/right hand side cannot be empty.")
    if (expression.replace(" ","")).count("x") == 0:
        raise SyntaxError("The expression must contain the variable x.")
    if any(left_side[-1] == avoid for avoid in ["-","+","*","/","%"]) or any(left_side[0] == avoid for avoid in ["*","/","%"]) or any(right_side[0] == avoid for avoid in ["*","/","%"]) or any(right_side[-1] == avoid for avoid in ["-","+","*","/","%"]):
        raise SyntaxError("The expression contains uncompleted operation.")
    if interval_end < interval_start:
        raise ValueError("The interval_start must be less than or equal to the interval_end.")
    if (deprived_start != None and deprived_end == None) or (deprived_start == None and deprived_end != None):
        raise ValueError("Deprived interval must have an start/end.")
    if deprived_start == deprived_end == None:
        pass
    elif deprived_start > deprived_end:
        raise ValueError("The deprived interval_start must be less than or equal to the deprived interval_end.")
    if deprived_values != None:
        if type(deprived_values) is not list:
            raise TypeError("Deprived_values type must be a list")
        else:
            if len(deprived_values) == 0:
                raise ValueError("Deprived values cannot be an empty list.")
            for value in deprived_values:
                if type(value) != float and type(value) != int:
                    raise TypeError("Deprived values of the list must be float or integer.")
    def func(x):
        return eval(left_side)-eval(right_side)
    test = 0
    try:
        func(test)
    except SyntaxError:
        raise SyntaxError("There was a problem in the giving expression.")
    except:
        pass
    R_solutions = []
    C_solutions = []
    R_result = []
    C_result = []
    status = True
    def solve_in_R():
        x = 0
        while status:
            try:
                f = func(x)
                while type(f) is complex:
                    x = randint(-100, 100)
                    f = func(x)
            except OverflowError or ValueError:
                x = randint(-10, 10)
                continue
            except ZeroDivisionError:
                x += 1
                continue
            except:
                if interval_start == float("-inf") and interval_end == float("inf"):
                    x = randint(-100, 100)
                elif interval_start == float("-inf") and interval_end != float("inf"):
                    x = randint(-100, int(interval_end))
                elif interval_start != float("-inf") and interval_end == float("inf"):
                    x = randint(int(interval_start), 100)
                else:
                    x = randint(int(interval_start), int(interval_end))
                continue
            if abs(f) < 1e-12:
                R_solutions.append(x)
                if interval_start == float("-inf") and interval_end == float("inf"):
                    x = randint(-100,100)
                elif interval_start == float("-inf") and interval_end != float("inf"):
                    x = randint(-100, int(interval_end))
                elif interval_start != float("-inf") and interval_end == float("inf"):
                    x = randint(int(interval_start), 100)
                else:
                    x = randint(int(interval_start),int(interval_end))
                continue
            try:
                f_prime = derivative(func, x)
                x -= f/f_prime
                
            except:
                x += 1
    def solve_in_C():
        x = 1 + 1j
        while status:
            try:
                f = func(x)
            except SyntaxError:
                raise SyntaxError("There is a problem in the expression.")
            except OverflowError or ValueError:
                x = randint(-10,10)+randint(-10, 10)*1j
                continue
            except ZeroDivisionError:
                x += 1 + 1j
                continue
            except:
                x = randint(-100, 100)+randint(-100, 100)*1j
                continue
            if abs(f) < 1e-12:
                C_solutions.append(x)
                x = randint(-100, 100)+randint(-100, 100)*1j
                continue
            try:
                f_prime = derivative(func, x)
                x -= f/f_prime
            except:
                x += 1 + 1j
    if real:
        Thread(target=solve_in_R).start()
        Thread(target=solve_in_R).start()
        Thread(target=solve_in_R).start()
        Thread(target=solve_in_R).start()
        Thread(target=solve_in_R).start()
    if cplx:
        Thread(target=solve_in_C).start()
        Thread(target=solve_in_C).start()
        Thread(target=solve_in_C).start()
        Thread(target=solve_in_C).start()
        Thread(target=solve_in_C).start()
    sleep(3)
    status = False
    eliminated_sol_from_R = []
    for sol in R_solutions:
        if type(sol) != complex:
            eliminated_sol_from_R.append(sol)
    R_solutions = eliminated_sol_from_R
    if len(R_solutions) != 0:
        R_solutions[0] = round(R_solutions[0],10)
        R_result.append(R_solutions[0])
        R_solutions.pop(0)
        for solution in R_solutions:
            if (not any(abs(solution - sol) < 1e-4 for sol in R_result)) and (solution - interval_start) >= -1e-6 and (interval_end - solution) >= -1e-6:
                if deprived_start != None:
                    if solution > deprived_end or solution < deprived_start:
                        if deprived_values != None:
                            if any((abs(solution - prv) < 1e-4) for prv in deprived_values):
                                continue
                            else:
                                solution = round(solution,10)
                                R_result.append(solution)
                        else:
                            solution = round(solution,10)
                            R_result.append(solution)
                else:
                    if deprived_values != None:
                        if all(abs(solution - prv) > 1e-4 for prv in deprived_values):
                            solution = round(solution,10)
                            R_result.append(solution)
                    else:
                        solution = round(solution,10)
                        R_result.append(solution)
        if (R_result[0] - interval_start) >= -1e-6 and (interval_end - R_result[0]) >= -1e-6:
            if deprived_start != None:
                    if R_result[0] > deprived_end or R_result[0] < deprived_start:
                        if deprived_values != None:
                            if any(abs(R_result[0] - prv) < 1e-4 for prv in deprived_values):
                                R_result.pop(0)
                    else:
                        R_result.pop(0)
            elif deprived_values != None:
                if any(abs(R_result[0] - prv) < 1e-4 for prv in deprived_values):
                    R_result.pop(0)
        else:
            R_result.pop(0)
        R_result = sorted(R_result)
        pos = []
        neg = []
        for sol in R_result:
            if sol >= 0:
                pos.append(sol)
            else:
                neg.append(sol)
        R_result = pos + neg
    if len(C_solutions) != 0:
        C_solutions[0] = round(C_solutions[0].real, 10) + round(C_solutions[0].imag, 10) * 1j
        if abs(C_solutions[0].imag) > 1e-4:
            C_result.append(C_solutions[0])
        C_solutions.pop(0)
        for solution in C_solutions:
            if not any(abs(solution - sol) < 1e-4 for sol in C_result):
                solution = round(solution.real, 10) + round(solution.imag, 10) * 1j
                if abs(solution.imag) > 1e-4:
                    C_result.append(solution)
    if max_solutions == None:
        result = R_result + C_result
    else:
        result = []
        for i in range(max_solutions):
            if i <= len(R_result)-1:
                result.append(R_result[i])
        for i in range(max_solutions):
            if i <= len(C_result)-1:
                result.append(C_result[i])
    return result

st.set_page_config(
    page_title="Aziz Bot",
    page_icon="img/favicon.png"
)

def main():
    st.title("Math equation solver")
    expression = st.text_input("Enter the expression: (Required)")
    cplx = False
    real = True
    if st.checkbox("Real", value=True):
        real = True
    else:
        real = False
    if st.checkbox("Complex"):
        cplx = True
    else:
        cplx = False
    max_solutions = st.number_input("Maximum amount of solutions: (Optional)", value=None ,min_value=0, max_value=None, step=1)
    col1, col2 = st.columns(2)
    with col1:
        i_s = st.text_input("Interval start: (Default -inf)")
        i_e = st.text_input("Interval end: (Default +inf)")
    with col2:
        d_s = st.text_input("Deprived interval start: (Optional)")
        d_e = st.text_input("Deprived interval end: (Optional)")
    d_v = st.text_input("Deprived values: (Values must be separated by a comma)")
    if st.button("Sumbit"):
        if i_s == "":
            interval_start = float("-inf")
        else:
            interval_start = eval(i_s)
        if i_e == "":
            interval_end = float("inf")
        else:
            interval_end = eval(i_e)
        if d_s == "":
            deprived_start = None
        else:
            deprived_start = eval(d_s)
        if d_e == "":
            deprived_end = None
        else:
            deprived_end = eval(d_e)
        if d_v == "":
            deprived_values = None
        else:
            d_v_list = d_v.split(",")
            deprived_values = []
            for value in d_v_list:
                deprived_values.append(eval(value))
        sol = equation_solver(expression, real=real, cplx=cplx, max_solutions=max_solutions, interval_start=interval_start, interval_end=interval_end, deprived_start=deprived_start, deprived_end=deprived_end, deprived_values=deprived_values)
        solve = st.success(sol)

        expression = (((((expression.replace(" ", "")).replace("^", "**")).replace("sqrt", "plot_sqrt")).replace("cbrt", "plot_cbrt")).replace("ln", "plot_ln")).replace("asin", "plot_asin")
        left_side = expression.split("=")[0]
        right_side = expression.split("=")[1]

        def c1(x):
            if left_side.count("x") == 0:
                return eval(left_side) + 0*x
            return eval(left_side)

        def c2(x):
            if right_side.count("x") == 0:
                return eval(right_side) + 0*x
            return eval(right_side)
        
        x = linspace(-15, 15, 400)

        fig, ax = plt.subplots(figsize=(5, 4))

        if left_side.count("j") == 0:
            ax.plot(x, c1(x), label=f"y = {((((left_side.replace("plot_sqrt", "sqrt")).replace("plot_cbrt", "cbrt")).replace("plot_ln", "ln")).replace("plot_asin", "asin")).replace("**", "^")}")
            if right_side.count("j") == 0:
                ax.plot(x, c2(x), label=f"y = {((((right_side.replace("plot_sqrt", "sqrt")).replace("plot_cbrt", "cbrt")).replace("plot_ln", "ln")).replace("plot_asin", "asin")).replace("**", "^")}")
            ax.set_facecolor("#131720")
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            ax.spines['left'].set_position('zero')
            ax.spines['bottom'].set_position('zero')
            ax.spines['left'].set_color('white')
            ax.spines['bottom'].set_color('white')
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
            ax.grid(True, which='both')
            ax.minorticks_on()
            ax.grid(True, which='major', color='#262730', linewidth=0.5)
            ax.grid(True, which='minor', color='#2d2e39', linewidth=0.5)
            plt.legend(loc="upper left")
            plt.xlim((-11, 11))
            plt.ylim((-11, 11))
            plt.tight_layout()
            plot = plt.savefig('plot.png')
            image = Image.open('plot.png')
            width, height = image.size
            left = 19
            top = 15
            right = width - 15
            bottom = height - 15
            image = image.crop((left, top, right, bottom))
            plot = st.image(image)
        elif right_side.count("j") == 0:
            ax.plot(x, c2(x), label=f"y = {(((right_side.replace("plot_sqrt", "sqrt")).replace("plot_cbrt", "cbrt")).replace("plot_ln", "ln")).replace("plot_asin", "asin")}")
            ax.set_facecolor("#131720")
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            ax.spines['left'].set_position('zero')
            ax.spines['bottom'].set_position('zero')
            ax.spines['left'].set_color('white')
            ax.spines['bottom'].set_color('white')
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
            ax.grid(True, which='both')
            ax.minorticks_on()
            ax.grid(True, which='major', color='#262730', linewidth=0.5)
            ax.grid(True, which='minor', color='#2d2e39', linewidth=0.5)
            plt.legend(loc="upper left")
            plt.xlim((-11, 11))
            plt.ylim((-11, 11))
            plt.tight_layout()
            plot = plt.savefig('plot.png')
            image = Image.open('plot.png')
            width, height = image.size
            left = 19
            top = 15
            right = width - 15
            bottom = height - 15
            image = image.crop((left, top, right, bottom))
            plot = st.image(image)

    data = {
        'Operation/Function': ["+","-","*","/","** or ^","%","//","sqrt(x)","cbrt(x)","sin(x, deg=False)","cos(x, deg=False)","tan(x, deg=False)","cot(x, deg=False)","sec(x, deg=False)","csc(x, deg=False)","asin(x)","acos(x)","atan(x)","acot(x)","asec(x)","acsc(x)","sinh(x)","cosh(x)","tanh(x)","coth(x)","sech(x)","csch(x)","floor(x)","ceil(x)","gcd(a, b)","lcm(a, b)","factorial(x)","integrate(integrand, a, b)","exp(x)","log(a, base=10)","ln(x)","derivative(func, value)"],
        'Name': ["Addition","Subtraction","Multiplication","Division","Exponentiation","Modulus","Floor division","Square root","Cubic root","Sine","Cosine","Tangent","Cotangent","Secant","Cosecant","arc Sine","arc Cosine","arc Tangent","arc Cotangent","arc Secant","arc Cosecant","Hyperbolic Sine","Hyperbolic Cosine","Hyperbolic Tangent","Hyperbolic Cotangent","Hyperbolic Secant","Hyperbolic Cosecant","Floor","Ceiling","Greatest Common Divisor","Least Common Multiple","Factorial","Integral of integrand (function type)","Exponential","Logarithm","Natural Logarithm","Derivative"]
    }
    st.table(data)

if __name__ == "__main__":
    main()