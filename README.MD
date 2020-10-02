# Practice neural nets 101
Digest of the Hacker's guide to neural nets from [Andrej Karpathy blog](https://karpathy.github.io/neuralnets/) consulted on sept 2020.  

## Single gate scenario

### Random Local Search
We want to increase the output of a single `forward multiply gate` defined as 
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
Our net is composed of 2 gates
* one `forward add gate` or `+` and
* one `forward multiply gate` or `*`  

From above, only considering `*`, we know that
* the derivative wrt `q` is `z` and
* the derivative wrt `z` is `q`
* thus the gradient being `(y,x)`  

Moreover, looking at `+` we know that
* the derivative wrt `x` is `1` and
* the derivative wrt `y` is `1`
* thus the gradient being `(1,1)`

### Backpropagation
Is "chain-rolling" the gradients of the entire net to get its final gradient value as, in a 2-gates net
```python
# 2-gates net
def forwardNet(x,y,z):
    q = forwardAddGate(x,y) # default = 3
    f = forwardMultiplyGate(q,z) # default = -12

    ## from * gate
    derivative_f_wrt_z = q # = 3
    derivative_f_wrt_q = z # = -4

    ## from + gate
    derivative_q_wrt_x = 1.0
    derivative_q_wrt_y = 1.0

    ## chain rule
    derivative_f_wrt_x = derivative_q_wrt_x * derivative_f_wrt_q # = -4
    derivative_f_wrt_y = derivative_q_wrt_y * derivative_f_wrt_q # = -4

    ## final gradient
    gradient_f_wrt_xyz = [
        derivative_f_wrt_x,
        derivative_f_wrt_y,
        derivative_f_wrt_z
    ]

    ## makes inputs converge
    step = 0.01
    x = x + step_size * derivative_f_wrt_x
    y = y + step_size * derivative_f_wrt_y
    z = z + step_size * derivative_f_wrt_z

    ## updates net
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
```
reaching output of `f = -11.59`, better than default `-12`.  

To be checked with numerical gradient as
```python
# numerical gradient on a 2-gates net
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
```
returning `TRUE` for `[-4, -4, 3]` as `x,y,z`.

## Single Neuron
Let's apply all this to a single 2-dimensional neuron with sigmoid normalizer that we will not over simplify, concatenating multiple gates into higher order ones through classes.

First we need to store the forward pass value and the backpropagated gradient into a unit as
```python
class Unit:
    def __init__(self, value, grad):
        self.value = value
        self.grad = grad
```  

Then processing the sigmoid function implies
* one `forward multiply gate` or `*` as seen before although here defined as a class for modularity puposes
```python
class MultiplyGate:
    # forward
    def forward(self, x, y):
        # stores x & y units and returns z
        self.x = x
        self.y = y
        self.z = Unit(x.value * y.value, 0.0)
        return z
    
    # backward
    def backward(self):
        # updates local gradients by chaining z with themselves
        self.x.grad = self.x.grad + self.y.value * self.z.grad
        self.y.grad = self.y.grad + self.x.value * self.z.grad
```
* then one `forward add gate` or `+` as well
```python
class AddGate:
    # forward
    def forward(self, x, y):
        self.x = x
        self.y = y
        self.z = Unit(x.value + y.value, 0.0)
        return z
    
    # backward
    def backward(self, x, y):
        self.x.grad = self.x.grad + 1 * self.z.grad
        self.y.grad = self.y.grad + 1 * self.z.grad
```
* and one `forward sigmoid gate` or `sig`
defined as
```python
class SigmoidGate:
    # support declaration
    import math

    # support expression
    def sig(self, x):
        return 1 / (1 + math.exp(-x))
    
    # forward
    def forward(self, x):
        self.x = x
        self.z = Unit(self.sig(self.x.value), 0.0)
        return self.z

    # backward
    def backward(self):
        s = self.sig(self.x.value)
        self.x.grad = self.x.grad + (s * (1 - s)) * self.z.grad
```  

Instantiate the units and gates to represent our situation
```python
# units
a = Unit(1.0,0.0)
b = Unit(2.0,0.0)
c = Unit(-3.0,0.0)
x = Unit(-1.0,0.0)
y = Unit(3.0,0.0)

# gates
multiply_gate_0 = MultiplyGate()
multiply_gate_1 = MultiplyGate()
add_gate_0 = AddGate()
add_gate_1 = AddGate()
sigmoid_gate = SigmoidGate()
```
Setup the forward pass of the neuron and run it
```python
def forwardNeuron():
    print("\nNeuron par classes :")
    ax = multiply_gate_0.forward(a,x)
    print("ax = ",type(ax),ax.value,ax.grad)
    by = multiply_gate_1.forward(b,y)
    print("by = ",type(by),by.value,by.grad)

    axbpy = add_gate_0.forward(ax,by)
    print("axbpy = ",type(axbpy),axbpy.value,axbpy.grad)
    axpbypc = add_gate_1.forward(axbpy,c)
    print("axpbypc = ",type(axpbypc),axpbypc.value,axpbypc.grad)

    s = sigmoid_gate.forward(axpbypc)
    print("s = ",type(s),s.value,s.grad)
    return (
        s.value,
        s.grad
    )
## run
forwardNeuron()
```