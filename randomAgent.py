import gym

env = gym.make('BipedalWalker-v3', render_mode="human")
env.reset()

observation = env.reset()

while True:
    env.render()
    print(observation)
    action = env.action_space.sample()
    
    observation,reward,done,trunc,info = env.step(action)
    
    if done:
        break

env.close()
