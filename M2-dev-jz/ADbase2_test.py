from ADbase2 import *




# variable x
x=AD(3)

y = x*2

print(y)


############################################
# define function i
i = 2*x + 3*x**2 + 2

# value of i
# g=2*3+3*3^2+2 = 35
print(float(i.val))

# jacobian of i by hand
# g'= 2+3*2*3=20

# jacobian by AD
print(float(i.der))


############################################

# define function g
g=6*x**2


# value of g
# 6*3^2=54
print(float(g.val))

# jacobian of g by hand
# g'= 2*6*x
# g'= 2*6*3=36

# jacobian by AD
print(float(g.der))






# ############################################

# # define function h as e^x
# h=x.exp()

# # value of g
# # e^3~20.0855
# print(float(h.val))

# # jacobian of h by hand
# # g'= e^x
# # g'= e^3~20.0855
# # jacobian by AD
# print(float(h.der))



# k=x.sin()

# # value of k
# # sin(3)~0.1411, 
# # NOTICE, here 3 are Radian, not numerical 3!!
# print(float(k.val))

# # jacobian of k by hand
# # k'= cos(x)
# # k'= cos(3)~-.9899
# # jacobian by AD
# print(float(k.der))




# j=x.sqrt()

# # value of k
# # 3^0.5
# print(float(j.val))

# # jacobian of k by hand
# # j'=0.5*x^-0.5
# print(float(j.der))