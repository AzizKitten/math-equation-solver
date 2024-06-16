import streamlit as st
from random import randint
from threading import Thread
from time import sleep
e = 2.718281828459045090795598298427648842334747314453125
pi = 3.141592653589793115997963468544185161590576171875

def sinh(x: float | complex) -> float | complex:
    """
    Return the hyperbolic sine of x.
    """
    return (e**x-e**(-x))/2

def cosh(x: float | complex) -> float | complex:
    """
    Return the hyperbolic cosine of x.
    """
    return (e**x+e**(-x))/2

def tanh(x: float | complex) -> float | complex:
    """
    Return the hyperbolic tangent of x.
    """
    return sinh(x)/cosh(x)

def coth(x: float | complex) -> float | complex:
    """
    Return the hyperbolic cotangent of x.
    """
    return cosh(x)/sinh(x)

def sech(x: float | complex) -> float | complex:
    """
    Return the hyperbolic secant of x.
    """
    return 1/cosh(x)

def csch(x: float | complex) -> float | complex:
    """
    Return the hyperbolic cosecant of x.
    """
    return 1/sinh(x)

def sqrt(x: float | complex) -> float | complex:
    """
    Return the square root of x.
    """
    if type(x) is not complex:
        if x <= 0:
            return (-x)**.5*1j
    return x**.5

def gcd(a:int,b:int) -> int:
    """
    Return the greatest common divisor of two integers a and b.
    """
    while b != 0:
        a, b = b, a % b
    return a

def lcm(a:int,b:int) -> int:
    """
    Return the least common multiple of two integers a and b.
    """
    return abs(a*b) // gcd(a,b)

def floor(x: float) -> int:
    """
    Return the floor of x, the largest integer less than or equal to x.
    """
    if x == int(x) or x >= 0:
        return int(x)
    return int(x)-1

def ceil(x: float) -> int:
    """
    Return the ceiling of x, the smallest integer greater than or equal to x.
    """
    if x == int(x) or x <= 0:
        return int(x)
    return int(x)+1

def exp(x: float | complex) -> float | complex:
    """
    Return the exponential function of x.
    """
    return e**x

def ln(x: float | complex) -> float | complex:
    """
    Return the natural logarithm of x.
    """
    if x == 0:
        raise ValueError("Value input must be different to 0.")
    if type(x) is not complex:
        if x > 0:
            if x == 1:
                return 0
            y = x
            while True:
                f = exp(y) - x
                f_prime = exp(y)
                if abs(f_prime) < 1e-6:
                    y += 1
                else:
                    y -= f/f_prime
                    if abs(f) < 1e-12:
                        return y
        y = abs(x)
        if abs(x) == 1:
            return pi*1j
        while True:
                f = exp(y) - abs(x)
                f_prime = exp(y)
                if abs(f_prime) < 1e-6:
                    y += 1
                else:
                    y -= f/f_prime
                    if abs(f) < 1e-12:
                        return y + pi*1j
    else:
        a = x.real
        b = x.imag
        if a == 0:
            return ln(b) + 1j*pi/2
        return ln(abs(x)) + 1j*atan(b/a)

def log(x: float, base=10) -> float | complex:
    """
    Return the logarithm of x in 'base'.

    Default base is 10 for decimal logarithm.
    """
    return(ln(x)/ln(base))

def integrate(integrand, lower_limit: float, upper_limit: float) -> float:
    """
    Return the result of the integral of 'integrand' from 'lower_limit' to 'upper_limit'.
    """
    n = 10000
    if upper_limit >= 5000:
        upper_limit = 5000
    if upper_limit >= 5000:
        upper_limit = -5000
    if lower_limit >= 5000:
        lower_limit = 5000
    if lower_limit >= 5000:
        lower_limit = -5000
    
    segment_width = (upper_limit - lower_limit) / n
    result = 0.5 * (integrand(lower_limit) + integrand(upper_limit))
    for i in range(1,n):
        x_i = lower_limit + i * segment_width
        result += integrand(x_i)
    result *= segment_width
    if result >= 1e10:
        return float('inf')
    elif result <= -1e10:
        return float('-inf')
    return result

def factorial(x: float) -> float:
    """
    Return the factorial of x where x ∉ {-1, -2, -3, ...}.
    """
    if int(x) == x:
        if x < 0:
            raise ValueError("Factorial of a negative integer is undefined.")
        else:
            ans = 1
            for i in range(1,int(x)+1):
                ans *= i
            return ans
    def integrand(t):
        return t**x * exp(-t)
    return integrate(integrand, 0, 100)

def sin(x: float | complex, deg=False) -> float | complex:
    """
    Return the sine of x.

    Measured in radians as defualt.
    """
    if deg:
        x = pi*(x/180)
    if type(x) is not complex:
        return ((e**(1j*x)-e**(1j*-x))/2j).real
    if abs(((e**(1j*x)-e**(1j*-x))/2j).imag) < 1e-6:
        return ((e**(1j*x)-e**(1j*-x))/2j).real
    return (e**(1j*x)-e**(1j*-x))/2j

def cos(x: float | complex, deg=False) -> float | complex:
    """
    Return the cosine of x.

    Measured in radians as defualt.
    """
    if deg:
        x = pi*(x/180)
    if type(x) is not complex:
        return ((e**(1j*x)+e**(1j*-x))/2).real
    if abs(((e**(1j*x)+e**(1j*-x))/2).imag) < 1e-6:
        return ((e**(1j*x)+e**(1j*-x))/2).real
    return (e**(1j*x)+e**(1j*-x))/2

def tan(x: float | complex, deg=False) -> float | complex:
    """
    Return the tangent of x.

    Measured in radians as defualt.
    """
    if deg:
        x = pi*(x/180)
    return sin(x)/cos(x)

def cot(x: float | complex, deg=False) -> float | complex:
    """
    Return the cotangent of x.

    Measured in radians as defualt.
    """
    if deg:
        x = pi*(x/180)
    return cos(x)/sin(x)

def sec(x:float | complex, deg=False) -> float | complex:
    """
    Return the secant of x.

    Measured in radians as defualt.
    """
    if deg:
        x = pi*(x/180)
    return 1/cos(x)

def csc(x: float | complex, deg=False) -> float | complex:
    """
    Return the cosecant of x.

    Measured in radians as defualt.
    """
    if deg:
        x = pi*(x/180)
    return 1/sin(x)

def asin(x: float) -> float:
    """
    Return the arc sine (measured in radians) of x.
    """
    if -1 <= x <= 1:
        if x == 1:
            return pi/2
        elif x == -1:
            return -(pi/2)
        else:
            def integrand(t):
                return 1/(1-t**2)**.5
            return integrate(integrand, 0, x)
    raise ValueError("Value input must be in [-1 .. 1]")

def acos(x: float) -> float:
    """
    Return the arc cosine (measured in randians) of x.
    """
    if -1 <= x <= 1:
        return pi/2 - asin(x)
    else:
        raise ValueError("Value input must be in [-1 .. 1]")
    
def atan(x: float) -> float:
    """
    Return the arc tangent (measured in radians) of x.
    """
    def integrand(t):
        return 1/(1+t**2)
    return integrate(integrand, 0, x)

def acot(x: float) -> float:
    """
    Return the arc cotangent (measured in radians) of x.
    """
    return atan(1/x)

def asec(x:float) -> float:
    """
    Return the arc secant (measured in radians) of x.
    """
    return acos(1/x)

def acsc(x:float) -> float:
    """
    Return the arc cosecant (measured in radians) of x.
    """
    return asin(1/x)

def cbrt(x: float | complex) -> float | complex:
    """
    Return the cubic root of x.
    """
    if type(x) is complex:
        a = x.real
        b = x.imag
        if a == 0:
            return cbrt(b)*-1j
        if b == 0:
            return cbrt(a)
        if a > 0 and b > 0:
            return cbrt(sqrt(a**2+b**2))*exp(1j*(atan(b/a))/3)
        if a > 0 and b < 0:
            return cbrt(sqrt(a**2+b**2))*exp(1j*(2*pi-atan(-b/a))/3)
        if a < 0 and b > 0:
            return cbrt(sqrt(a**2+b**2))*exp(1j*(pi-atan(b/-a))/3)
        return cbrt(sqrt(a**2+b**2))*exp(1j*(pi+atan(b/a))/3)
    if x == 0:
        return 0.0
    y = x
    while True:
        f = y**3-x
        if abs(f) < 1e-12:
            return float(y)
        f_prime = 3*y**2
        if f_prime == 0:
            y += 1
        else:
            y -= f/f_prime

def derivative(func: 'function', value: float | complex) -> float | complex:
    """
    Return the derivative of a function 'func' at a specific value.
    """
    h=1e-10
    ans=(func(value+h)-func(value))/h
    return ans

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
        raise ValueError("The expression must contain only one '='.")
    left_side = expression.split("=")[0].replace(" ", "")
    right_side = expression.split("=")[1].replace(" ","")
    if len(left_side) == left_side.count(" ") or len(right_side) == right_side.count(" "):
        raise ValueError("Left/right hand side cannot be empty.")
    if (expression.replace(" ","")).count("x") == 0:
        raise ValueError("The expression must contain the variable x.")
    if any(left_side[-1] == avoid for avoid in ["-","+","*","/","%"]) or any(left_side[0] == avoid for avoid in ["*","/","%"]) or any(right_side[0] == avoid for avoid in ["*","/","%"]) or any(right_side[-1] == avoid for avoid in ["-","+","*","/","%"]):
        raise ValueError("The expression contains uncompleted operation.")
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
    page_title="Math Bot",
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
        i_s = st.text_input("Interval start: (Leave it empty for -infinity)")
        i_e = st.text_input("Interval end: (leave it empty for +infinity)")
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
        solve = equation_solver(expression, real=real, cplx=cplx, max_solutions=max_solutions, interval_start=interval_start, interval_end=interval_end, deprived_start=deprived_start, deprived_end=deprived_end, deprived_values=deprived_values)
        solve = st.success(solve)
    data = {
        'Operation/Function': ["+","-","*","/","**","%","//","sqrt(x)","cbrt(x)","sin(x, deg=False)","cos(x, deg=False)","tan(x, deg=False)","cot(x, deg=False)","sec(x, deg=False)","csc(x, deg=False)","asin(x)","acos(x)","atan(x)","acot(x)","asec(x)","acsc(x)","sinh(x)","cosh(x)","tanh(x)","coth(x)","sech(x)","csch(x)","floor(x)","ceil(x)","gcd(a, b)","lcm(a, b)","factorial(x)","integrate(integrand, a, b)","exp(x)","log(a, base=10)","ln(x)","derivative(func, value)"],
        'Name': ["Addition","Subtraction","Multiplication","Division","Exponentiation","Modulus","Floor division","Square root","Cubic root","Sine","Cosine","Tangent","Cotangent","Secant","Cosecant","arc Sine","arc Cosine","arc Tangent","arc Cotangent","arc Secant","arc Cosecant","Hyperbolic Sine","Hyperbolic Cosine","Hyperbolic Tangent","Hyperbolic Cotangent","Hyperbolic Secant","Hyperbolic Cosecant","Floor","Ceiling","Greatest Common Divisor","Least Common Multiple","Factorial","Integral of integrand (function type)","Exponential","Logarithm","Natural Logarithm","Derivative"]
    }
    st.table(data)

if __name__ == "__main__":
    main()