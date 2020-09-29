# Single gate scenario

## forward multiply gate
def forwardMultiplyGate(x,y):
    return x * y

## default values
x = -2.0
y = 3.0

# Random Local Search
import random

def random_tweaking(x,y):
    tweak_amount = 0.01
    best_out = -100.0
    best_x = x
    best_y = y
    for k in range(100):
        x_try = x + tweak_amount * (random.random() * 2 - 1)
        y_try = y + tweak_amount * (random.random() * 2 - 1)
        out = forwardMultiplyGate(x_try,y_try)
        if out > best_out:
            best_out = out
            best_x = x_try
            best_y = y_try
    ## returns bests x, y & output
    return (
        best_x,
        best_y,
        best_out
        )
## run
print(
    "\nWith random tweaking :",
    random_tweaking(x,y),
    sep = "\n"
    )

# Numerical Gradient
def numerical_gradient(x,y):
    h = 0.0001
    out = forwardMultiplyGate(x,y)

    ## deriving x
    xph = x + h
    out2 = forwardMultiplyGate(xph,y)
    x_derivative = (out2 - out) / h

    ## deriving y
    yph = y + h
    out3 = forwardMultiplyGate(x,yph)
    y_derivative = (out3 - out) / h

    ## computing numerical gradient components
    step = 0.01
    x = x + step * x_derivative
    y = y + step * y_derivative
    out_new = forwardMultiplyGate(x,y)

    ## returns 1-step gradient convergent result
    return (
        x,
        y,
        out_new
    )
## run
print(
    "\nWith numerical gradient :",
    numerical_gradient(x,y),
    sep = "\n"
    )

# Analytic Gradient
def analytic_gradient(x,y):
    ## case specific shortcut
    x_gradient = y
    y_gradient = x

    ## computes gradient components
    step = 0.01
    x = x + step * x_gradient
    y = y + step * y_gradient
    out_new = forwardMultiplyGate(x,y)

    ## returns 1-step gradient convergent result
    return (
        x,
        y,
        out_new
    )
## run
print(
    "\nWith analytic gradient :",
    analytic_gradient(x,y),
    sep = "\n"
    )

# Multiple gate scenario

## new add gate
def forwardAddGate(x,y):
    return x + y

## default values
x = -2
y = 5
z = -4
out_default = -12

## backpropagation on a 2-gates net
def forwardNet(x,y,z):
    q = forwardAddGate(x,y) # default = 3
    f = forwardMultiplyGate(q,z) # default = -12

    ### from * gate
    derivative_f_wrt_z = q # = 3
    derivative_f_wrt_q = z # = -4

    ### from + gate
    derivative_q_wrt_x = 1.0
    derivative_q_wrt_y = 1.0

    ### chain rule
    derivative_f_wrt_x = derivative_q_wrt_x * derivative_f_wrt_q # = -4
    derivative_f_wrt_y = derivative_q_wrt_y * derivative_f_wrt_q # = -4

    ### final gradient
    gradient_f_wrt_xyz = [
        derivative_f_wrt_x,
        derivative_f_wrt_y,
        derivative_f_wrt_z
    ]

    ### makes inputs converge
    step = 0.01
    x = x + step * derivative_f_wrt_x
    y = y + step * derivative_f_wrt_y
    z = z + step * derivative_f_wrt_z

    ### updates net
    q = forwardAddGate(x,y)
    f = forwardMultiplyGate(q,z)

    return (
        x,
        y,
        z,
        q,
        f,
        gradient_f_wrt_xyz
    )
## run
print(
    "\nBackpropagating in a 2-gates net :",
    forwardNet(x,y,z),
    sep = "\n"
    )

def numerical_gradient_multiple_gates(x,y,z):
    ### step
    h = 0.0001

    ### bench values by our net
    gradient_hat = forwardNet(x,y,z)[-1]
    hat = forwardNet(x,y,z)[-2]

    ### numerical check
    x_derivative = (forwardNet(x+h,y,z)[-2] - hat) / h
    y_derivative = (forwardNet(x,y+h,z)[-2] - hat) / h
    z_derivative = (forwardNet(x,y,z+h)[-2] - hat) / h
    check = [
        x_derivative,
        y_derivative,
        z_derivative
    ]

    ### diagnostic
    return [round(n) for n in gradient_hat] == [round(n) for n in check]
## run
print(
    "\nNumerical check :",
    numerical_gradient_multiple_gates(x,y,z)
    )