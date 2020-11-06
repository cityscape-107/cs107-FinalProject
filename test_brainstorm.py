import math
from brainstorm import Forward, sin, ln

x = Forward(0.5)
z = x**2 + sin(x) + ln(x)
print(z.grad())
