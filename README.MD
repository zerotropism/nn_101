# Practice neural nets 101
Digest in Python of the Hacker's guide to neural nets from [Andrej Karpathy blog](https://karpathy.github.io/neuralnets/) consulted on sept 2020.  

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
    def forward(self,x,y):
        # stores x & y units and returns their product z
        self.x = x
        self.y = y
        self.z = Unit(self.x.value * self.y.value, 0.0)
        return self.z

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
    def forward(self,x,y):
        # stores x & y units and returns their sum z
        self.x = x
        self.y = y
        self.z = Unit(self.x.value + self.y.value, 0.0)
        return self.z
    
    # backward
    def backward(self):
    # updates local gradients by incrementing with z
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

Instantiate the units with example values and gates to represent our situation
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
    ax = multiply_gate_0.forward(a,x)
    by = multiply_gate_1.forward(b,y)

    axbpy = add_gate_0.forward(ax,by)
    axpbypc = add_gate_1.forward(axbpy,c)

    s = sigmoid_gate.forward(axpbypc)
    return s
## run
forwardNeuron() # = 0.8808
```
* then set the gradient to 1 in order to allow the gradient chain to start
* and backpropagate as
```python
# backpropagation
s.grad = 1.0
sigmoid_gate.backward()
add_gate_1.backward()
add_gate_0.backward()
multiply_gate_1.backward()
multiply_gate_0.backward()
```
The units values will be updated with chained gradients times the step size and rerun the forward pass with updated values as
```python
step = 0.01
a.value = a.value + step * a.grad
b.value = b.value + step * b.grad
c.value = c.value + step * c.grad
x.value = x.value + step * x.grad
y.value = y.value + step * y.grad

forwardNeuron() # = .08825
```
The forward pass will return an output value of `0.8808` and once backpropagated it will return `0.8825`, thus showing a convergence towards a value increase.  

To be checked with numerical gradient as
```python
# numerical gradient on single neuron
def numerical_gradient_single_neuron(a,b,c,x,y):
    # fast forward
    return 1 / (1 + math.exp(-(a*x+b*y+c)))

# step
h = 0.0001

# example inputs
a,b,c,x,y = 1,2,-3,-1,3

# bench values
hat = numerical_gradient_single_neuron(a,b,c,x,y)

# numerical gradients
a_grad = (numerical_gradient_single_neuron(a+h,b,c,x,y) - hat) / h
b_grad = (numerical_gradient_single_neuron(a,b+h,c,x,y) - hat) / h
c_grad = (numerical_gradient_single_neuron(a,b,c+h,x,y) - hat) / h
x_grad = (numerical_gradient_single_neuron(a,b,c,x+h,y) - hat) / h
y_grad = (numerical_gradient_single_neuron(a,b,c,x,y+h) - hat) / h
```
with
```python
# computed analytically through the single neuron forward pass and backpropagation
analytic = [
    a.grad,
    b.grad,
    c.grad,
    x.grad,
    y.grad,
]

# computed numerically
numerical = [
    a_grad,
    b_grad,
    c_grad,
    x_grad,
    y_grad
]
```
returning
* `analytic = [-0.105, 0.315, 0.105, 0.105, 0.21]` and
* `numerical = [-0.105, 0.315, 0.105, 0.105, 0.21]` thus confirming the results.

## More on Backpropagation
Focusing on the sole backpropagation process, let's put our code structure away for a moment and consider the values `a, b, c & x` as well as their gradients `da, db, dc, dx` as integers. The complexity of the examples will go crecendo.  

#### Two numbers example 
With two numbers, the `*` gate gives
```python
# the * gate gives
x = a * b
# the gradients computes
da = b * dx
db = a * dx
```
assuming `dx` is given or equals `1` by default. Because its local gradient always returns `1.0`, the `+` gate on its end gives
```python
# gate
x = a + b
# gradients
da = 1.0 * dx
db = 1.0 * dx
```
where `1.0` is the local gradient and the `*` operator is the chain rule.  

#### Three numbers example
With three numbers example nothing changes, the `+` gate gives
```python
# gate
x = a + b + c
# gradients
da = 1.0 * dx
db = 1.0 * dx
dc = 1.0 * dx
```
and combining the gates yields
```python
# gate
x = a * b + c
# gradients
da = b * dx
db = a * dx
dc = 1.0 * dx
```

#### Compelete 2-dimensional neuron example
Along this simplification, setting up our neuron, including its sigmoid normalizer sets as
```python
# forward
q = a*x + b*y + c
f = sig(q)

# backward
df = 1.0 # by default
dq = (f * (1-f)) * df

da = x * dq
dx = a * dq
dy = b * dq
db = y * dq
dc = 1.0 * dq
```

#### More complex examples
When complex cases show up, split the expression into more manageable, simpler ones. For a multiplication
```python
# the entire expression 
x = math.pow((a * b + c) * d), 2)

# can be split in 3 simpler gates
x1 = a * b + c
x2 = x1 * d
x = x2 * x2

# allowing easier gradients computations (= backprop equations)
dx2 = 2 * x2 * dx
dd = x1 * dx2
dx1 = d * dx2
da = b * dx1
db = a * dx1
dc = 1.0 * dx1
```  
For a division
```python
# the expression
x = (a + b)/(c + d)

# decomposed in gates
x1 = a + b
x2 = c + d
x3 = 1.0 / x2
x = x1 * x3

# gradients (always in reverse order for backprop)
dx1 = x3 * dx
dx3 = x1 * dx
dx2 = (-1.0 / x2²) * dx3
da = 1.0 * dx1
db = 1.0 * dx1
dc = 1.0 * dx2
dd = 1.0 * dx2
```  

## Machine Learning
Forward pass & backward pass mechanisms applied to Machine Learning.

### Binary Classification
Let's have a `N = 6, D = 2` vectors outputting a `-1` or `+1` label as
```
vector -> label
---------------
[1.2, 0.7] -> +1
[-0.3, 0.5] -> -1
[-3, -1] -> +1
[0.1, 1.0] -> -1
[3.0, 1.1] -> -1
[2.1, -3] -> +1
```
The objective is to learn a function that takes a 2-dimensional vector and predicts the label, dropping the sigmoid normalizer for simplification purposes as
* `x` & `y` the inputs (2D vectors),
* `a, b, c` the parameters  
and we then proceed to a `Stochastic Gradient Descent` as training scheme consisting of
* random input values
* compute outputs (predictions)
* compare to provided labels
* tug on circuit in correct direction
* backpropagate
* compute with updated values
* iterate

### Support Vector Machine
Based on the approach from first chapter, the complete circuit class should resemble this
```python
## circuit class
class Circuit:
    # gates declarations
    multiply_gate_0 = MultiplyGate()
    multiply_gate_1 = MultiplyGate()
    add_gate_0 = AddGate()
    add_gate_1 = AddGate()

    # forward process
    def forward(self,x,y,a,b,c):
        self.ax = multiply_gate_0.forward(a,x)
        self.by = multiply_gate_1.forward(b,y)
        self.axbpy = add_gate_0.forward(self.ax,self.by)
        self.axpbypc = add_gate_1.forward(self.axbpy,c)
        return self.axpbypc

    # backward process
    def backward(self,gradient_top):
        self.axpbypc.grad = gradient_top
        add_gate_1.backward()           # sets gradient in axpby and c
        add_gate_0.backward()           # sets gradient in ax and by
        multiply_gate_1.backward()      # sets gradient in b and y
        multiply_gate_0.backward()      # sets gradient in a and x
```
which simply computes `a*x + b*y + c` as well as the gradient.  

Now a SVM class should resemble the following
```python
class SVM:
    a = Unit(1.0, 0.0)
    b = Unit(-2.0, 0.0)
    c = Unit(-1.0, 0.0)
    circuit = Circuit()
    unit_out = Unit(0 ,0)

    def forward(self,x,y):
        self.unit_out = self.circuit.forward(x,y,self.a,self.b,self.c)
        return self.unit_out
    
    def backward(self,label):           # label is +1 or -1
        # reset grad values to start chaining
        self.a.grad = 0.0
        self.b.grad = 0.0
        self.c.grad = 0.0
        self.pull = 0.0
        if label == 1 and self.unit_out.value < 1:
            self.pull = 1
        elif label == -1 and self.unit_out.value > -1:
            self.pull = -1
        self.circuit.backward(self.pull)
        self.a.grad = self.a.grad - self.a.value
        self.b.grad = self.b.grad - self.b.value

    def parameterUpdate(self):
        step = 0.01
        self.a.value = self.a.value + step + self.a.grad
        self.b.value = self.b.value + step + self.b.grad
        self.c.value = self.b.value + step + self.c.grad

    def learnFrom(self,x,y,label):
        self.forward(x,y)
        self.backward(label)
        self.parameterUpdate()
```
Which we will train with stochastic gradient descent as
```python
### compute SGD
data = []
data.extend(
    [[1.2, 0.7]]
    + [[-0.3, -0.5]]
    + [[3.0, 0.1]]
    + [[-0.1, -1.0]]
    + [[-1.0, 1.1]]
    + [[2.1, -3.0]]
    )
labels = [] 
labels.extend(
    [1]
    + [-1]
    + [1]
    + [-1]
    + [-1]
    + [1]
    )
svm = SVM()

#### assess classification accuracy
def evalTrainingAccuracy():
    num_correct = 0
    for i in range(len(data)):
        x = Unit(data[i][0], 0.0)
        y = Unit(data[i][1], 0.0)
        true_label = labels[i]
        predicted_label = 1 if svm.forward(x,y).value > 0 else -1
        if predicted_label == true_label:
            num_correct += 1
    return num_correct / len(data)

import math
import random

#### learning loop
for k in range(400):
    # pick random data point
    i = math.floor(random.random() * len(data))
    x = Unit(data[i][0], 0.0)
    y = Unit(data[i][1], 0.0)
    label = labels[i]
    svm.learnFrom(x,y,label)
    if k % 25 == 0:
        print("training accuracy at iteration n°", k, ":", evalTrainingAccuracy())
```
Which outputs
```
training accuracy at iteration n° 0 : 0.5
training accuracy at iteration n° 25 : 0.8333333333333334
training accuracy at iteration n° 50 : 0.6666666666666666
training accuracy at iteration n° 75 : 0.5
training accuracy at iteration n° 100 : 0.5
training accuracy at iteration n° 125 : 0.6666666666666666
training accuracy at iteration n° 150 : 0.6666666666666666
training accuracy at iteration n° 175 : 0.6666666666666666
training accuracy at iteration n° 200 : 0.6666666666666666
training accuracy at iteration n° 225 : 0.5
training accuracy at iteration n° 250 : 0.5
training accuracy at iteration n° 275 : 0.6666666666666666
training accuracy at iteration n° 300 : 0.8333333333333334
training accuracy at iteration n° 325 : 0.6666666666666666
training accuracy at iteration n° 350 : 0.8333333333333334
training accuracy at iteration n° 375 : 0.6666666666666666
```