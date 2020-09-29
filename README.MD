# Practice neural nets 101
Digest of the Hacker's guide to neural nets from [Andrej Karpathy blog](https://karpathy.github.io/neuralnets/) consulted on sept 2020.  

## Single gate scenario

### Random Local Search
We want to increase the output of a `single forward multiply gate` defined as 
```python
# forward multiply gate
def forwardMultiplyGate(x,y):
    return x * y
```
with default input values set as 
```python
# default values
x = -2.0
y = 3.0
out_default = forwardMultiplyGate(x,y) # default = -6
```
tweaked over 100 iterations as  
```python
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
```
reaching a best output of `best_out = -5.95`.

### Numerical Gradient
Same objective, but different method, here by deriving the inputs as
```python
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

    ## computes numerical gradient components
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
```
reaching an output of `out_new = -5.87`.

### Analytic Gradient
Given the single `forward multiply gate` and the two sole inputs we have, we can state that
* the derivative wrt `x` is `y` and
* the derivative wrt `y` is `x`
thus allowing us to compute the gradient much more efficiently as
```python
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
```
reaching same output of `out_new = -5.87` with lighter computing.  

The gradient was computed by forwarding through the gate
* 100 times with random tweaking,
* twice the number of inputs times with numerical gradient and
* only once with analytic gradient.

## Multiple gates scenario
We now add a `forward add gate` defined as
```python
# new add gate
def forwardAddGate(x,y):
    return x + y
```
which now composes our net of two gates such as
```python
# 2-gates net
def forwardNet(x,y,z):
    q = forwardAddGate(x,y)
    f = forwardMultiplyGate(q,z)
    return f
```
with default input values set as
```python
# default values
x = -2
y = 5
z = -4
out_default = forwardNet(x,y,z) # default = -12
```

