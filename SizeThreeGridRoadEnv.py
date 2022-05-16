# openai gym related import statements.
# building first a simpler environment that works with stablebaselines
# then adding the rllib related nuances in a different environment.
import gym
from gym import spaces
import numpy as np
import random

# Implementation of the 3x3 road network for MultiAgent RL.
class SizeThreeGridRoadEnv(gym.Env):
    # Defining the Driving Agent with 
    class DriverAgent():
        def __init__(self, name, gas):
            self.name = name
            self.gas = gas

    # Different possible world configurations.
    world_one = np.array([[1, 0, 0],
                  [3, 0, 2],
                  [0, 0, 4]])
    world_two = np.rot90(world_one)
    world_three = np.rot90(world_two)
    world_four = np.rot90(world_three)
    # Even the initial world configuration should be different.
    prob = random.uniform(0, 1)
    # Default value assignment below.
    world = world_one
    if prob > 0.25 and prob <= 0.25:
        world = world_two
    elif prob > 0.5 and prob <= 0.75:
        world = world_three
    elif prob > 0.75 and prob <= 1:
        world = world_four
    def __init__(self):
        self.world_start = world
        # Adding five actions for the environment.
        # 0: up, 1: right, 2: down, 3: left, 4: stay, 5: pick, 6: drop
        self.action_space = spaces.Discrete(7)
        shape_0 = np.size(self.world_start, 0)
        shape_1 = np.size(self.world_start, 1)
        self.observation_space = spaces.Box(low=0,
                                            high=256,
                                            shape=(shape_0 + 1, shape_1),
                                            dtype=np.int16)
        self.reward_range = (-10, 1)
        self.current_episode = 0
        self.success_episode = []
        self.agent_one = DriverAgent(1,4)
        self.agent_two = DriverAgent(2,4)

    def reset(self):
        # Game like formulation, each player agent moves one step at a time.
        self.current_player = agent_one
        # P means the game is playable, W means somenone wins, L someone loses.
        self.state = 'P'
        self.current_step = 0
        self.max_step = 30 # agent can choose not move as an alternate choice.
        # Selecting a world at random to function with.
        # Even the initial world configuration should be different.
        prob = random.uniform(0, 1)
        if prob > 0.25 and prob <= 0.25:
            self.world_start = world_two
        elif prob > 0.5 and prob <= 0.75:
            self.world_start = world_three
        elif prob > 0.75 and prob <= 1:
            self.world_start = world_four
        elif prob < 0.25:
            self.world_start = world_one
        self.world = np.copy(self.world_start)
        # no exploration_prize and bonus_reward as per my design.
        return self._next_observation()
    
    def _next_observation(self):
        obs = self.world
        data_to_add = [0] * np.size(self.world, 1)
        data_to_add[0] = self.current_player.name # adding current player's label.
        obs = np.append(obs, [data_to_add], axis=0)
        return obs


    def _take_action(self, action):
        # Checking logic corresponding to the agent's concatenated number
        # is also needed to be added here.
        # the conditional check when carrying the package.
        agent_name = self.current_player.name
        if self.current_player.name > 2:
            name_item_list = []
            num = self.current_player.name
            while (num != 0):
                temp = num % 10
                name_item_list.append(temp)
                num = num // 10
            agent_name = name_item_list[len(name_item_list)-1]

        current_pos = np.where(self.world == agent_name)
        # the current agent must have gas in it.
        if self.current_player.gas > 0:


            if action == 0:
                next_pos = (current_pos[0] - 1, current_pos[1]) # Agent moving upwards.

                if next_pos[0] >= 0 and int(self.world[next_pos]) == 0:
                    self.world[next_pos] = self.current_player.name
                    self.world[current_pos] = 0
                    # Reducing the agent's gas by 1.
                    self.current_player.gas = self.current_player.gas - 1

                elif next_pos[0] >= 0 and int(self.world[next_pos]) in (1, 2):
                    pass # Two Agents can't be at the same place.

                elif next_pos[0] >= 0 and int(self.world[next_pos] == 3):
                    # Existing w/ in the world '10 or 20' + 3 as per this logic.
                    self.world[next_pos] = self.current_player.name*10 + 3 
                    self.world[current_pos] = 0
                    # Reducing the agent's gas by 1.
                    self.current_player.gas = self.current_player.gas - 1

                elif next_pos[0] >= 0 and int(self.world[next_pos] == 4):
                    self.world[next_pos] = self.current_player.name*10 + 4 # like for player 234, for example.
                    self.world[current_pos] = 0
                    self.state = 'W'
                    # Reducing the agent's gas by 1.
                    self.current_player.gas = self.current_player.gas - 1


            elif action == 1:
                next_pos = (current_pos[0], current_pos[1] + 1)
                limit = np.size(self.world, 1)

                if next_pos[1] < limit and int(self.world[next_pos]) == 0:
                    self.world[next_pos] = self.current_player.name
                    self.world[current_pos] = 0
                    # Reducing the agent's gas by 1.
                    self.current_player.gas = self.current_player.gas - 1

                elif next_pos[1] < limit and int(self.world[next_pos]) in (1, 2):
                    pass # Two Agents can't be at the same place.

                elif next_pos[1] < limit and (int(self.world[next_pos]) == 3):
                    # Existing w/ in the world '10 or 20' + 3 as per this logic.
                    self.world[next_pos] = self.current_player.name*10 + 3 
                    self.world[current_pos] = 0
                    # Reducing the agent's gas by 1.
                    self.current_player.gas = self.current_player.gas - 1

                elif next_pos[1] < limit and int(self.world[next_pos] == 4):
                    self.world[next_pos] = self.current_player.name*10 + 4 # like for player 234, for example.
                    self.world[current_pos] = 0
                    self.state = 'W'
                    # Reducing the agent's gas by 1.
                    self.current_player.gas = self.current_player.gas - 1


            elif action == 2:
                next_pos = (current_pos[0] + 1, current_pos[1])
                limit = np.size(self.world, 0)

                if next_pos[0] < limit and int(self.world[next_pos]) == 0:
                    self.world[next_pos] = self.current_player.name
                    self.world[current_pos] = 0
                    # Reducing the agent's gas by 1.
                    self.current_player.gas = self.current_player.gas - 1

                elif next_pos[0] < limit and int(self.world[next_pos]) in (1, 2):
                    pass # Two Agents can't be at the same place.

                elif next_pos[0] < limit and (int(self.world[next_pos]) == 3):
                    # Existing w/ in the world '10 or 20' + 3 as per this logic.
                    self.world[next_pos] = self.current_player.name*10 + 3 
                    self.world[current_pos] = 0
                    # Reducing the agent's gas by 1.
                    self.current_player.gas = self.current_player.gas - 1

                elif next_pos[0] < limit and int(self.world[next_pos] == 4):
                    self.world[next_pos] = self.current_player.name*10 + 4 # like for player 234, for example.
                    self.world[current_pos] = 0
                    self.state = 'W'
                    # Reducing the agent's gas by 1.
                    self.current_player.gas = self.current_player.gas - 1

            elif action == 3:
                next_pos = (current_pos[0], current_pos[1] - 1)

                if next_pos[1] >= 0 and int(self.world[next_pos]) == 0:
                    self.world[next_pos] = self.current_player.name
                    self.world[current_pos] = 0
                    # Reducing the agent's gas by 1.
                    self.current_player.gas = self.current_player.gas - 1

                elif next_pos[1] >= 0 and int(self.world[next_pos]) in (1, 2):
                    pass # Two Agents can't be at the same place.

                elif next_pos[1] >= 0 and (int(self.world[next_pos]) == 3):
                    # Existing w/ in the world '10 or 20' + 3 as per this logic.
                    self.world[next_pos] = self.current_player.name*10 + 3 
                    self.world[current_pos] = 0
                    # Reducing the agent's gas by 1.
                    self.current_player.gas = self.current_player.gas - 1

                elif next_pos[1] >= 0 and int(self.world[next_pos] == 4):
                    self.world[next_pos] = self.current_player.name*10 + 4 # like for player 234, for example.
                    self.world[current_pos] = 0
                    self.state = 'W'
                    # Reducing the agent's gas by 1.
                    self.current_player.gas = self.current_player.gas - 1


            # Newly added logic based on three new possible actions
            elif action == 4:
                pass # Corresponding agent selects to not move at their chance.
            elif action == 5:
                # Agent selects to pick the package. It is practically equivalent to
                # changing the name of the agent permanently & making location empty.
                if world[current_pos] == self.current_player.name*10 + 3:
                    self.current_player.name = self.current_player.name*10 + 3
                    self.world[current_pos] = 0
            elif action == 6:
                # A logic to check concated agent's name and dimensions
                # i.e. dropping should only if the agent is carrying the package.
                if int(self.current_player.name) % 10 == 3:
                    if world[current_pos] == 0:
                        self.world[current_pos] = 3
                    elif world[current_pos] == 4: # Added as extra case, functionally shouldn't be triggered.
                        self.world[current_pos] = 34
                        self.state = 'W'
        else:
            self.state = 'L'


    def step(self, action):
        self._take_action(action)
        self.current_step += 1
        print(self.world)

        if self.state == "W":
            reward = 1
            done = True
        elif self.state == 'L':
            reward = -10
            done = True
        elif self.state == 'P':
            reward = 0 # sparse reward encoding, only rewarded when episode ends.
            done = False

        if self.current_step >= self.max_step:
            print(f'New episode number {self.current_episode + 1}')
            done = True

        # agents object used to identify agent properties.
        if self.current_player.name == 1 or self.current_player.name == 13  : # State: 134, not achievable.
            self.current_player = agent_two
        elif self.current_player.name == 2 or self.current_player.name == 23 or self.current_player.name == 234 : # At 234, the state changes to 'W', episode end.
            self.current_player = agent_one

        if done:
            self.render_episode(self.state)
            self.current_episode += 1

        obs = self._next_observation()

        return obs, reward, done, {'state': self.state}

    def render_episode(self, win_or_lose):
        # Storing the rendered episodes in a file.
        self.success_episode.append(
            'Success' if win_or_lose == 'W' else 'Failure')
        file = open('render.txt', 'a')
        file.write('----------------------------\n')
        file.write(f'Episode number {self.current_episode}\n')
        file.write(
            f'{self.success_episode[-1]} in {self.current_step} steps\n')
        file.close()
