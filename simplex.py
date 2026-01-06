import sympy as sp

x, y = sp.symbols('x y')
f = x**3 + 3*y*x**2 - 2*y**2
df_dx = sp.diff(f, x)
df_dy = sp.diff(f, y)
print("Partial derivative with respect to x:")
sp.pprint(df_dx)
print("Partial derivative with respect to y:")
sp.pprint(df_dy)
H = sp.hessian(f, (x, y))
print("Hessian matrix:")
sp.pprint(H)