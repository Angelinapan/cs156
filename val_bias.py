import math
from random import uniform

def ran_var():
	e1 = uniform(0, 1)
	e2 = uniform(0, 1)
	if e1 < e2:
		return e1, e2, e1
	else:
		return e1, e2, e2

tot_e1 = 0
tot_e2 = 0
tot_e = 0
N = 100000
for i in range(N):
	e1, e2, e = ran_var()
	tot_e1 += e1
	tot_e2 += e2
	tot_e += e
print(tot_e1/N, tot_e2/N, tot_e/N)