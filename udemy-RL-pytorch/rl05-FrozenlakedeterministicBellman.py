import gym
import matplotlib.pyplot as plt
import time
import statistics as stats
import torch
from numba import prange

from gym.envs.registration import register

register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False }
)

env = gym.make("FrozenLakeNotSlippery-v0")

no_of_action = env.action_space.n
no_of_states = env.observation_space.n
Q = torch.zeros([no_of_states,no_of_action])
gamma = 0.9
num_episodes = 1000
steps_list = []
rewards_total = []
time1 = time.time()
for i in prange(num_episodes):
    state = env.reset()
    steps = 0
    while True:
        random_values = Q[state] + torch.randn(1,no_of_action)/10000
        #action = env.action_space.sample()
        action = torch.max(random_values,1)[1][0].item()
        #print(action)
        new_state, reward, done, info = env.step(action)
        Q[state,action] = reward + gamma*torch.max(Q[new_state])
        state = new_state
        steps +=1
        #time.sleep(0.4)
        #env.render()
        #print(new_state)
        #print(info)

        if done:
            steps_list.append(steps)
            rewards_total.append(reward)
            #print("episodes finished after %i steps" %steps)
            break
time2 = time.time()
print(time2-time1)
print("percentage of episodes finished successfully: {0}".format(sum(rewards_total)/num_episodes))
print("average number of steps = %f" %(stats.mean(steps_list)))
plt.plot(steps_list)
plt.show()
plt.plot(rewards_total)
plt.show()
