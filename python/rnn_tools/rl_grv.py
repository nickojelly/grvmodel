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
        state = np.concatenate([self.price_preds[step], self.model_preds[step],self.prices[step]])
        # state = np.array(self.prices[step])

        return state

    def get_reward(self, action, step):
        dog, size, direction = action
        # dog = action
        print(f"{dog=},{size=},{direction=},{step=}")
        # print(f"{dog=}")

        # Define bet sizes
        bet_sizes = [10, 50, 100]  # replace with your actual bet sizes

        # Check if the action is "do nothing"
        if dog == -1 and size == -1 and direction == -1:
            return 0

        # Get the result and price for the selected dog
        favorite = np.argmax(self.prices[step])
        winner = np.argmax(self.results[step])
        result = self.results[step][dog]
        price = 1/self.prices[step][dog]
        if np.isnan(price) or np.isinf(price):
            price = 1

        

        if result == 1:  # the dog won
            # reward = 1
            reward = bet_sizes[size]*price - bet_sizes[size]
        else:  # the dog lost
            reward =  - bet_sizes[size]

        # Calculate the reward
        # if direction == 0:  # back
        #     if result == 1:  # the dog won
        #         reward = 1
        #         # reward = bet_sizes[size] * price - bet_sizes[size]
        #     else:  # the dog lost
        #         reward = 0
        #         # reward = -bet_sizes[size]
        # # else:  # lay
        #     # if result == 1:  # the dog won
        #         # reward = -bet_sizes[size] * price
        #     # else:  # the dog lost
        #         # reward = bet_sizes[size]
        # else:
        #     reward = 0
        # print(f"{reward=}")
        # print(f"{result=},{price=},{favorite=},{reward=}\n")
        return reward

class GreyhoundRacingEnv(gym.Env):
    def __init__(self, data):
        super(GreyhoundRacingEnv, self).__init__()

        self.data:RaceData = data
        self.current_step = 0

        # Define action and observation space
        # self.action_space = spaces.Discrete(48)
        self.action_space = spaces.Discrete(24)
        # self.action_space = spaces.Discrete(8)
        low = np.zeros(24)
        high = np.ones(24)  
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def decode_action(self, action):
        # print(f"{action=}")
        dog = action // 6
        size = (action % 6) // 2
        direction = action % 2
        direction = 0
        return dog, size, direction

    def step(self, action):
        # Execute one time step within the environment

        reward = self.calculate_reward(action, self.state)
        self.reward = reward

        self.current_step += 1
        state = self._get_state()

        done = self.current_step >= len(self.data) - 1

        return state, reward, done, {}

        
    def _get_state(self):
        state = self.data.get_state(self.current_step)
        self.state = state
        self.max_index = np.argmax(state)

        return state

    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_step = 0
        state = self.data.get_state(self.current_step)
        self.state = state
        return state

    def calculate_reward(self, action, state):
        # Decode the action
        action = self.decode_action(action)
        # Calculate the reward based on the action and the state
        reward = self.data.get_reward(action, self.current_step)
        # print(reward)
        self.reward = reward
        return reward