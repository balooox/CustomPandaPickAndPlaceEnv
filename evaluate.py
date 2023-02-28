import sys
import gym
import panda_gym
import custom_env
from stable_baselines3.common.evaluation import evaluate_policy
from sb3_contrib import TQC

argv_len = len(sys.argv)

if argv_len == 1:
    print("Usage: python ./evaluate env algo file")
    print("env: the gym environment")
    print("algo: training algorithm, can be TQC or SAC")
    print("file: path to .zip file of trained model")
    exit(-1)
elif argv_len == 2 or argv_len == 3:
    print("More parameters are required")
    print("Usage: python ./evaluate env algo file")
    print("Help: execute python ./enjoy for more information")
    print(-1)
elif argv_len > 4:
    print("Too many parameters")
    print("Usage: python ./evaluate env algo file")
    print("Help: execute python ./enjoy for more information")
    exit(-1)

env_id = sys.argv[1]

if env_id != "PandaPickAndPlaceAndThrow-v1" and env_id != "PandaPickAndPlaceAndMove-v1":
    print("Wrong environment id!")
    print("Please use PandaPickAndPlaceAndThrow-v1 or PandaPickAndPlaceAndMove-v1")
    exit(-1)

algo = sys.argv[2]

if algo != "TQC" and algo != "SAC":
    print("Wrong training algorithm")
    print("Parameter algo can only be 'TQC' or 'SAC'")
    exit(-1)

path_to_zip = sys.argv[3]

env = gym.make(env_id, render=False)

model = None
if algo == "TQC":
    model = TQC.load(path_to_zip, env=env)
elif algo == "SAC":
    print("Not tested yes!")
    exit(-1)

episodes = 100

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=episodes, deterministic=True)
print(f"### mean_reward={mean_reward:.2f} +/- {std_reward}")
