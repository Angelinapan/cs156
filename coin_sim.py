import numpy as np
import math


def calcFrequency(coin):
	num_heads = 0
	for c in coin:
		if c == 'H':
			num_heads += 1
	return float(num_heads) / 10.0


def runFlips():
	toss_results = ['H', 'T']
	coins = []
	for i in range(1000):
		coins.append(list(np.random.randint(2, size=10)))
		indices = coins.pop()
		coin_tosses = []
		for i in indices:
			coin_tosses.append(toss_results[i])
		coins.append(coin_tosses)

	c_1 = coins[0]
	v_1 = calcFrequency(c_1)

	c_rand = coins[np.random.randint(1000)]
	v_rand = calcFrequency(c_rand)

	min_freq = 1.0
	c_min = []
	for coin in coins:
		freq = calcFrequency(coin)
		if freq < min_freq:
			min_freq = freq
			c_min = coin
	v_min = calcFrequency(c_min)
	return v_1, v_rand, v_min


def main():
	s = [0.0, 0.0, 0.0]
	count2 = [0.0, 0.0, 0.0]
	count3 = [0.0, 0.0, 0.0]
	count4 = [0.0, 0.0, 0.0]
	for i in range(100000):
		if i % 1000 == 0:
			print(i)
		v = runFlips()
		for j in range(3):
			if math.fabs(v[j] - 0.5) > 0.2:
				count2[j] += 1.0
			if math.fabs(v[j] - 0.5) > 0.3:
				count3[j] += 1.0
			if math.fabs(v[j] - 0.5) > 0.4:
				count4[j] += 1.0
			s[j] += v[j]
	
	# Average values:
	print(s[0] / float(100000))		# v_1
	print(s[1] / float(100000))		# v_rand
	print(s[2] / float(100000))		# v_min
	print("")

	# P[|v-u| > e]
	# e = 0.2:
	print(count2[0] / float(100000))	# v_1
	print(count2[1] / float(100000))	# v_rand
	print(count2[2] / float(100000))	# v_min
	print("")
	# e = 0.3:
	print(count3[0] / float(100000))	# v_1
	print(count3[1] / float(100000))	# v_rand
	print(count3[2] / float(100000))	# v_min
	print("")
	# e = 0.4:
	print(count4[0] / float(100000))	# v_1
	print(count4[1] / float(100000))	# v_rand
	print(count4[2] / float(100000))	# v_min
	print("")

if __name__ == "__main__":
	main()