# SET 5:
# Using gradient descent to find the minimum of the nonlinear error surface:
# 			E(u,v) = (u*e^v - 2*v*e^(-u))^2
# We start at point (u,v) = (1,1) and use learning rate = 0.1

from math import exp 

# Partial of E(u,v) wrt u.
def partial_u(u, v):
	return 2 * (u * exp(v) - 2 * v * exp(-1*u)) * (exp(v) + 2 * v * exp(-1*u))

# Partial of E(u,v) wrt v.
def partial_v(u, v):
	return 2 * (u * exp(v) - 2 * v * exp(-1*u)) * (u * exp(v) - 2 * exp(-1*u))

# Error surface function.
def E(u, v):
	return (u * exp(v) - 2 * v * exp(-1*u))**2

# Gradient descent algorithm.
def grad_descent(n, u, v, termination):
	error = E(u, v)
	iterations = 0
	while error > termination:
		temp_u = n * partial_u(u, v)
		temp_v = n * partial_v(u, v)
		u -= temp_u
		v -= temp_v
		error = E(u, v)
		iterations += 1
	print(u, v)
	return iterations

# Now, trying to minimize error using coordinate descent.
def coord_descent(n, u, v):
	error = E(u, v)
	for i in range(15):
		u -= n * partial_u(u, v)
		v -= n * partial_v(u, v)
		error = E(u, v)
	return error

def main():
	# Comparing gradient vs. coordinate descent.
	print(grad_descent(0.1, 1.0, 1.0, 10.0**(-14)))
	print(coord_descent(0.1, 1.0, 1.0))

if __name__ == '__main__':
	main()