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

# Simple Neuron

class Unit:
    def __init__(self, value, grad):
        self.value = value
        self.grad = grad

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

class SigmoidGate:
    # support expression
    def sig(self,x):
        import math
        return 1 / (1 + math.exp(-x))

    # forward
    def forward(self,x):
        # stores unit and returns z
        self.x = x
        self.z = Unit(self.sig(self.x.value), 0.0)
        return self.z

    # backward
    def backward(self):
        s = self.sig(self.x.value)
        self.x.grad = self.x.grad + (s * (1 - s)) * self.z.grad

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

def forwardNeuron():
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
    return s
## run
print("\nneuron forward pass :")
s = forwardNeuron()

# backprop
s.grad = 1.0
sigmoid_gate.backward()
add_gate_1.backward()
add_gate_0.backward()
multiply_gate_1.backward()
multiply_gate_0.backward()

## update values with chained gradients times the step size
step = 0.01
a.value = a.value + step * a.grad
b.value = b.value + step * b.grad
c.value = c.value + step * c.grad
x.value = x.value + step * x.grad
y.value = y.value + step * y.grad

print("\nneuron backward pass :")
forwardNeuron()

## isolate analytic gradients
analytic = [
    a.grad,
    b.grad,
    c.grad,
    x.grad,
    y.grad,
]

## numerical gradient on single neuron
def numerical_gradient_single_neuron(a,b,c,x,y):
    # fast forward
    import math
    return 1 / (1 + math.exp(-(a*x+b*y+c)))

## step
h = 0.0001

## example inputs
a,b,c,x,y = 1,2,-3,-1,3

## bench values
hat = numerical_gradient_single_neuron(a,b,c,x,y)

## numerical gradients
a_grad = (numerical_gradient_single_neuron(a+h,b,c,x,y) - hat) / h
b_grad = (numerical_gradient_single_neuron(a,b+h,c,x,y) - hat) / h
c_grad = (numerical_gradient_single_neuron(a,b,c+h,x,y) - hat) / h
x_grad = (numerical_gradient_single_neuron(a,b,c,x+h,y) - hat) / h
y_grad = (numerical_gradient_single_neuron(a,b,c,x,y+h) - hat) / h

numerical = [
    a_grad,
    b_grad,
    c_grad,
    x_grad,
    y_grad
]

print("\nanalytic =", [round(n,3) for n in analytic])
print("numerical =", [round(n,3) for n in numerical])

# # More on Backpropagation
# ## multiplication
# ### the entire expression
# import math
# x = math.pow((a * b + c) * d, 2)

# ### can be split in 3 simpler gates
# x1 = a * b + c
# x2 = x1 * d
# x = x2 * x2

# ### allowing easier gradients computations (= backprop equations)
# dx2 = 2 * x2 * dx
# dd = x1 * dx2
# dx1 = d * dx2
# da = b * dx1
# db = a * dx1
# dc = 1.0 * dx1

# ## division
# ### the expression
# x = (a + b)/(c + d)

# ### decomposed in gates
# x1 = a + b
# x2 = c + d
# x3 = 1.0 / x2
# x = x1 * x3

# ### gradients (always in reverse order for backprop)
# dx1 = x3 * dx
# dx3 = x1 * dx
# dx2 = (-1.0 / x2**2) * dx3
# da = 1.0 * dx1
# db = 1.0 * dx1
# dc = 1.0 * dx2
# dd = 1.0 * dx2

# SVM
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

## svm class
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