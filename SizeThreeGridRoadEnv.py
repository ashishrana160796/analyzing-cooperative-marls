# Implementation of the 3x3 road network for MultiAgent RL.
class SizeThreeGridRoadEnv(gym.Env):
    # Defining the Driving Agent with name and gas values.
    class DriverAgent():
        def __init__(self, name, gas, package):
            self.name = name
            self.gas = gas
            self.package = package

    # Defining different possible world configurations.
    world_one = np.array([[1, 0, 0],
                  [3, 0, 2],
                  [0, 0, 4]])
    world_two = np.rot90(world_one)
    world_three = np.rot90(world_two)
    world_four = np.rot90(world_three)
    # Even the initial world configuration is defined to be different.
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
        self.world_start = world # This 'world_start', if reset() is called, never gets used.
        # Adding five actions for the environment.
        # 0: up, 1: right, 2: down, 3: left, 4: stay/pass, 5: pick, 6: drop
        self.action_space = spaces.Discrete(7)
        shape_0 = np.size(self.world_start, 0)
        shape_1 = np.size(self.world_start, 1)
        self.observation_space = spaces.Box(low=0,
                                            high=4,
                                            shape=(shape_0 + 1, shape_1),
                                            dtype=np.int16)
        self.reward_range = (-10, 1)
        self.current_episode = 0
        self.success_episode = []
        # Defining the driver agents in the environment.
        self.agent_one = DriverAgent(1,4,0) # 3 integer value, when carrying package.
        self.agent_two = DriverAgent(2,4,0) # 3 integer value, when carrying package.

    def reset(self):
        # Game like formulation, each player agent moves one step at a time.
        self.agent_one = DriverAgent(1,4,0) # Instantiating agent 1 again.
        self.agent_two = DriverAgent(2,4,0) # Instantiating agent 2 again.
        self.current_player = self.agent_one
        # 'P' means the game is playable, 'W' means delivered, 'L' means no delivery.
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
        self.world = np.copy(self.world_start) # The self.world can be different from intial world.
        # no exploration_prize and bonus_reward as per my design.
        return self._next_observation()
    
    def _next_observation(self):
        obs = self.world
        data_to_add = [0] * np.size(self.world, 1)
        data_to_add[0] = self.current_player.name # adding current player's label in the observation.
        obs = np.append(obs, [data_to_add], axis=0)
        return obs


    def _take_action(self, action):
        # Agent's name is matched to the array entries for index identification.
        # 'current_player.name' should be updated alongside the array values.
        current_pos = np.where(self.world == self.current_player.name)
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
                    self.world[next_pos] = self.current_player.name
                    self.current_player.package = 3 # package is also hidden now from other agent.
                    self.world[current_pos] = 0
                    # Reducing the agent's gas by 1.
                    self.current_player.gas = self.current_player.gas - 1

                elif next_pos[0] >= 0 and int(self.world[next_pos] == 4):
                    self.world[next_pos] = self.current_player.name*10 + 4 # like for player 24, for example.
                    self.world[current_pos] = 0
                    self.state = 'W' # and the episode, should end at that moment.
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
                    self.world[next_pos] = self.current_player.name
                    self.current_player.package = 3 # package is also hidden now from other agent.
                    self.world[current_pos] = 0
                    # Reducing the agent's gas by 1.
                    self.current_player.gas = self.current_player.gas - 1

                elif next_pos[1] < limit and int(self.world[next_pos] == 4):
                    # 4, i.e. delivery position should only be accessed when agent has the package.
                    if self.current_player.package == 3:
                        self.world[next_pos] = 34 # like for player 34, for example.
                        self.world[current_pos] = 0
                        self.state = 'W'
                        # Reducing the agent's gas by 1.
                        self.current_player.gas = self.current_player.gas - 1
                    # else, the action stays ineffective in nature.


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
                    self.world[next_pos] = self.current_player.name
                    self.current_player.package = 3 # package is also hidden now from other agent.
                    self.world[current_pos] = 0
                    # Reducing the agent's gas by 1.
                    self.current_player.gas = self.current_player.gas - 1

                elif next_pos[0] < limit and int(self.world[next_pos] == 4):
                    # 4, i.e. delivery position should only be accessed when agent has the package.
                    if self.current_player.package == 3:
                        self.world[next_pos] = 34 # like for player 34, for example.
                        self.world[current_pos] = 0
                        self.state = 'W'
                        # Reducing the agent's gas by 1.
                        self.current_player.gas = self.current_player.gas - 1
                    # else, the action stays ineffective in nature.

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
                    self.world[next_pos] = self.current_player.name
                    self.current_player.package = 3 # package is also hidden now from other agent.
                    self.world[current_pos] = 0
                    # Reducing the agent's gas by 1.
                    self.current_player.gas = self.current_player.gas - 1

                elif next_pos[1] >= 0 and int(self.world[next_pos] == 4):
                    # 4, i.e. delivery position should only be accessed when agent has the package.
                    if self.current_player.package == 3:
                        self.world[next_pos] = 34 # like for player 34, for example.
                        self.world[current_pos] = 0
                        self.state = 'W'
                        # Reducing the agent's gas by 1.
                        self.current_player.gas = self.current_player.gas - 1
                    # else, the action stays ineffective in nature.

            # Newly added logic based on three new possible actions
            elif action == 4:
                pass # Corresponding agent selects to not move at their chance.
            elif action == 5: # If agent is over the package, It has to pick it up.
                # Agent can choose to drop the package, if it is loaded with it.
                # After, dropping the package the agent should dissappear.
                if self.current_player.package == 3:
                    if world[current_pos] == 0:
                        self.world[current_pos] = 3
                    elif world[current_pos] == 4: # Added as extra case, functionally shouldn't be triggered.
                        self.world[current_pos] = 34
                        self.state = 'W'
        else:
            # Player 1's gas is supposed to go empty first.
            # Therefore, upon having empty gas tank player should be allowed to
            # drop the package in the environment and disappear from the location.
            if self.current_player.package == 3:
                self.world[current_pos] = self.current_player.package
            else
                self.world[current_pos] = 0 # If gas is finished, agent should dissappear.



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
        if self.current_player.name == 1:
            self.current_player = agent_two
        elif self.current_player.name == 2:
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
