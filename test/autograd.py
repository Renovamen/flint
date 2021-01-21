import sys
sys.path.append('/Users/zou/Renovamen/Developing/tinyark/')

from tinyark import Tensor

a = Tensor(2.0, requires_grad = True)
b = Tensor(2.0, requires_grad = True)

c = a * b
d = - 2 * c

d.backward()

print(d.data, a.grad)
