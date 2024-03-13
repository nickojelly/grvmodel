import gym
from gym import spaces
import numpy as np

class RaceData:
    def __init__(self, dogs,prices, price_preds, model_preds,results):
        self.dogs = dogs
        self.prices = prices
        self.results = results
        self.price_preds = price_preds
        self.model_preds = model_preds
        
    def __len__(self):
        # Return the number of steps in the race
        return len(self.dogs)
    

    def get_state(self, step):
        # Return the state of the race at the given step
        state = np.array(self.prices[step])

        return state

    def get_reward(self, action, step):
        # print(f"State in reward: {self.results[step]}")
        dog = action
        favorite = np.argmax(self.prices[step])
        winner = np.argmax(self.results[step])
        result = self.results[step][dog]


        if result == 1:  # the dog won
            reward = 1
        else:  # the dog lost
            reward = 0
        # print(f"step: {step}")
        # print(f"Dog: {dog} Reward: {reward}, Favorite: {favorite},Winner: {winner}, Result: {result}\n")
        

        return reward

class GreyhoundRacingEnv(gym.Env):
    def __init__(self, data):
        super(GreyhoundRacingEnv, self).__init__()

        self.data:RaceData = data
        self.current_step = 0

        self.action_space = spaces.Discrete(8)
        low = np.zeros(8)
        high = np.ones(8)  
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)


    def step(self, action):
        # Execute one time step within the environment
        
        # print(self.current_step)
        reward = 1 if action == self.max_index else 0
        self.reward = reward
        self.action = action
        # print(f"State in step: {self.state}")
        # print(f"Action: {action}, self.max_index: {self.max_index}, reward: {reward}")
        reward = self.calculate_reward(action, self.state)
        # print(f"Reward: {reward}")

        reward = self.reward
        self.current_step += 1
        state = self._get_state()
        
        done = self.current_step >= len(self.data) - 1

        return state, reward, done, {}
    
    def _get_state(self):
        state = self.data.get_state(self.current_step)
        self.state = state
        # print(f"{state=}")
        self.max_index = np.argmax(state)
        # print(self.max_index)
        return state

    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_step = 0

        state = self._get_state()
        self.state = state
        return state

    def calculate_reward(self, action, state):
        # Decode the action
        reward = self.data.get_reward(action, self.current_step)
        # print(reward)
        self.action = action
        self.reward = reward
        return reward
    
class MaxValueEnv(gym.Env):
    def __init__(self):
        super(MaxValueEnv, self).__init__()

        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32)
        self.seed()
        self.rewards = []

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def step(self, action):
        reward = 1 if action == self.max_index else 0
        self.state = self._get_state()
        done = False
        self.rewards.append(reward)

        if len(self.rewards) == 100:
            done = True
            print(f"Mean reward: {np.mean(self.rewards)}")
            print(f"Max reward: {np.max(self.rewards)}")
            print(f"Min reward: {np.min(self.rewards)}")
            print(f"Std dev reward: {np.std(self.rewards)}")
        
        return self.state, reward, done, {}

    def reset(self):
        self.state = self._get_state()
        self.rewards = []
        return self.state

    def _get_state(self):
        state = self.np_random.random(8)
        self.max_index = np.argmax(state)
        return state