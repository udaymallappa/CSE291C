import numpy as np
import matplotlib.pyplot as plt
import math
import random
from matplotlib.pyplot import cm




#inputs
u1 = 0.2
u2 = 0.8

epsilon = 0.01
time = 90000
alpha = 400
print("Mean values are :",u1," and " , u2)


##def float_range(start, stop, step):
##  while start < stop:
##	yield float(start)
##	start += decimal.Decimal(step)




def genExplore(u):
	temp = random.random()
	if temp <= u:
		return 1
	else:
		return 0


def genSample(u):
	temp = random.random()
	if temp <= u:
		return 1
	else:
		return 0




def bestArm(p,q):
	return max(p,q)




bu = bestArm(u1,u2)


color=iter(cm.rainbow(np.linspace(0,1,10)))

for epsilon in np.arange(0.01,0.2,0.05):

	#list of regret, and number of iterations
	regret = []
	
	#running mean values
	ru1 = 0
	ru2 = 0
	#running count
	r1 = 0
	r2 = 0
	acc_reward = 0
	acc_opt_reward = 0
	Reg =[]
	Rew =[]

	print('Epsilon value is: ', epsilon)
	for t in range(time):
		flag_explore = genExplore(epsilon)
		if flag_explore == 1:
					## explore option enabled, so choose the arm with lower running mean
			if ru1 < ru2:
				sample = genSample(u1)
				ru1 = (ru1 * r1 + sample) * 1.0 / (r1 + 1)
				r1 = r1 + 1
				acc_opt_reward += genSample(bu)
			else:
				sample = genSample(u2)
				ru2 = (ru2 * r2 + sample) * 1.0 / (r2 + 1)
				r2 = r2 + 1
				acc_opt_reward += genSample(bu)
		else:
					# no exploration, just pick the best arm by exploiting it
			if ru1 > ru2:
				sample = genSample(u1)
				ru1 = (ru1 * r1 + sample) * 1.0 / (r1 + 1)
				r1 = r1 + 1
				acc_opt_reward += genSample(bu)
			else:
				sample = genSample(u2)
				ru2 = (ru2 * r2 + sample) * 1.0 / (r2 + 1)
				r2 = r2 + 1
				acc_opt_reward += genSample(bu)
	
		acc_reward += sample
		Rew.append(acc_reward)
		acc_regret = acc_opt_reward - (acc_reward * 1.0)
		Reg.append(abs(acc_regret))
	c=next(color)	
	plt.plot(Reg,color=c)
#plt.legend(('epsilon = 0.1', 'epsilon = 0.01', 'epsilon = 0.001', 'epsilon = 0.2'),loc='upper right')
plt.ylabel('Acc Regret')
plt.xlabel('Number of rounds')
#plt.title('Mean 1 = 0.5, Mean 2 = 0.6')
#plt.show()


#plt.clf()
Reg_UCB = [] 
Rew_UCB = []
ru1 = 0
ru2 = 0
r1 = 0
r2 = 0
acc_reward = 0
acc_opt_reward = 0
t1 = 1
t2 = 1
cucb = 4
ru1 = genSample(u1)
ru2 = genSample(u2)

for t in range(time):

	u1_ucb = ru1 + math.sqrt( cucb * math.log(t+1) / t1 )
	u2_ucb = ru2 + math.sqrt( cucb * math.log(t+1) / t2 )
	if u1_ucb > u2_ucb:
		sample = genSample(u1)
		ru1 = (t1 * ru1 + sample) * 1.0 / (t1 + 1)
		t1 += 1
		acc_opt_reward += genSample(bu)
	else:
		sample = genSample(u2)
		ru2 = (t2 * ru2 + sample) * 1.0 / (t2 + 1)
		t2 += 1
		acc_opt_reward += genSample(bu)

	acc_reward += sample
	ucb_reg = acc_opt_reward - (acc_reward * 1.0)
	Reg_UCB.append(abs(ucb_reg))
	Rew_UCB.append(abs(acc_reward))

plt.plot(Reg_UCB,color='red',label='UCB')
plt.legend(('epsilon = 0.1', 'epsilon = 0.01', 'epsilon = 0.001', 'epsilon = 0.2', 'UCB'),loc='upper right')
#plt.legend('UCB')
plt.show()
