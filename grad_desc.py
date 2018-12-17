from math import exp 

def partial_u(u, v):
	return 2 * (u * exp(v) - 2 * v * exp(-1*u)) * (exp(v) + 2 * v * exp(-1*u))

def partial_v(u, v):
	return 2 * (u * exp(v) - 2 * v * exp(-1*u)) * (u * exp(v) - 2 * exp(-1*u))

def E(u, v):
	return (u * exp(v) - 2 * v * exp(-1*u))**2

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

def coord_descent(n, u, v):
	error = E(u, v)
	for i in range(15):
		u -= n * partial_u(u, v)
		v -= n * partial_v(u, v)
		error = E(u, v)
	return error

def main():
	print(grad_descent(0.1, 1.0, 1.0, 10.0**(-14)))
	print(coord_descent(0.1, 1.0, 1.0))

if __name__ == '__main__':
	main()