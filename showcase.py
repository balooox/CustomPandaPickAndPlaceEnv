import sys
import time
import gym
import custom_env

argv_len = len(sys.argv)

if argv_len == 1:
    print("Usage: python ./showcase env")
    print("env: the gym environment")
    exit(-1)
elif argv_len > 2:
    print("Too many parameters")
    print("Usage: python ./showcase env")
    print("Help: execute python ./showcase for more information")
    exit(-1)

env_id = sys.argv[1]

if env_id != "PandaPickAndPlaceAndThrow-v1" and env_id != "PandaPickAndPlaceAndMove-v1":
    print("Wrong environment id!")
    print("Please use PandaPickAndPlaceAndThrow-v1 or PandaPickAndPlaceAndMove-v1")
    exit(-1)


env = gym.make(env_id, render=True)

info = env.reset()
i = 0

while True:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)

    if i % 800 == 0:
        env.reset()

    if done:
        env.reset()

    time.sleep(0.05)
    i += 1
