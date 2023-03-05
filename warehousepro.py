import gym
from gym import spaces
import numpy as np
from gym.utils import seeding # random seed control 위해 import

import pyglet
import time

import os
from stable_baselines.common.vec_env import VecEnv, sync_envs_normalization, DummyVecEnv
from typing import Union, List, Optional, Tuple

from stable_baselines import DQN, PPO2
from stable_baselines.common.callbacks import EvalCallback, BaseCallback
# from stable_baselines.common.evaluation import evaluate_policy

def evaluate_policy_Warehouse(
    model,
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    return_episode_rewards: bool = False,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    This is made to work only with one env.

    :param model: (BaseRLModel) The RL agent you want to evaluate.
    :param env: (gym.Env or VecEnv) The gym environment. In the case of a ``VecEnv``
        this must contain only one environment.
    :param n_eval_episodes: (int) Number of episode to evaluate the agent
    :param deterministic: (bool) Whether to use deterministic or stochastic actions
    :param render: (bool) Whether to render the environment or not
    :param return_episode_rewards: (Optional[float]) If True, a list of reward per episode
        will be returned instead of the mean.
    :return: (float, float) Mean reward per episode, std of reward per episode
        returns ([float], [int]) when ``return_episode_rewards`` is True
    """
    if isinstance(env, VecEnv):
        assert env.num_envs == 1, "You must pass only one environment when using this function"

    epi_rewards_undiscounted, epi_rewards_discounted, epi_success_moves, epi_collisions, episode_lengths = [], [], [], [], []
    for i in range(n_eval_episodes):
        # Avoid double reset, as VecEnv are reset automatically
        if not isinstance(env, VecEnv) or i == 0:
            obs = env.reset()
        done, state = False, None
        epi_reward_undiscounted = 0.0
        epi_reward_discounted = 0.0
        epi_success = 0
        epi_collision = 0
        episode_length = 0
        num_steps = 0
        while not done:
            action, state = model.predict(obs, state=state, deterministic=deterministic)
            obs, reward, done, _info = env.step(action)
            epi_reward_undiscounted += reward
            epi_reward_discounted += np.power(model.gamma, num_steps) * reward
            num_steps += 1

            if _info[0].get('success'):
                epi_success += 1
            if _info[0].get('collision'):
                epi_collision += 1

            episode_length += 1
            if render:
                env.render()
        epi_rewards_undiscounted.append(epi_reward_undiscounted)
        epi_rewards_discounted.append(epi_reward_discounted)
        epi_success_moves.append(epi_success)
        epi_collisions.append(epi_collision)
        episode_lengths.append(episode_length)
    mean_discounted_reward = np.mean(epi_rewards_discounted)
    std_discounted_reward = np.std(epi_rewards_discounted)
    if return_episode_rewards:
        return epi_rewards_undiscounted, epi_rewards_discounted, epi_success_moves, epi_collisions, episode_lengths
    return mean_discounted_reward, std_discounted_reward

class EvalCallback_Warehouse(EvalCallback):
    def __init__(self, eval_env: Union[gym.Env, VecEnv],
                 callback_on_new_best: Optional[BaseCallback] = None,
                 n_eval_episodes: int = 5,
                 eval_freq: int = 10000,
                 log_path: str = None,
                 best_model_save_path: str = None,
                 deterministic: bool = True,
                 render: bool = False,
                 verbose: int = 1):
        super(EvalCallback_Warehouse, self).__init__(eval_env,
                 callback_on_new_best,
                 n_eval_episodes,
                 eval_freq,
                 log_path,
                 best_model_save_path,
                 deterministic,
                 render,
                 verbose)
        self.results_undiscounted = []
        self.results_discounted = []
        self.results_success = []
        self.results_collision = []

    def _on_step(self) -> bool:

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            sync_envs_normalization(self.training_env, self.eval_env)

            epi_rewards_undiscounted, epi_rewards_discounted, epi_success_moves, epi_collisions, episode_lengths \
                = evaluate_policy_Warehouse(self.model, self.eval_env,
                                            n_eval_episodes=self.n_eval_episodes,
                                            render=self.render,
                                            deterministic=self.deterministic,
                                            return_episode_rewards=True)


            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.results_undiscounted.append(epi_rewards_undiscounted)
                self.results_discounted.append(epi_rewards_discounted)
                self.results_success.append(epi_success_moves)
                self.results_collision.append(epi_collisions)
                self.evaluations_length.append(episode_lengths)
                np.savez(self.log_path, timesteps=self.evaluations_timesteps,
                         results_undiscounted=self.results_undiscounted,
                         results_discounted=self.results_discounted,
                         results_success=self.results_success,
                         ep_lengths=self.evaluations_length)

            mean_reward_undiscounted, std_reward_undiscounted = np.mean(epi_rewards_undiscounted), np.std(epi_rewards_undiscounted)
            mean_reward_discounted, std_reward_discounted = np.mean(epi_rewards_discounted), np.std(epi_rewards_discounted)
            mean_success, std_success = np.mean(epi_success_moves), np.std(epi_success_moves)
            mean_collision, std_collision = np.mean(epi_collisions), np.std(epi_collisions)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            # Keep track of the last evaluation, useful for classes that derive from this callback
            self.last_mean_reward = mean_reward_discounted

            if self.verbose > 0:
                print("Eval num_timesteps={}, "
                      "episode_discounted_reward={:.2f} +/- {:.2f}".format(self.num_timesteps, mean_reward_discounted, std_reward_discounted),
                      "episode_undiscounted_reward={:.2f} +/- {:.2f}".format(mean_reward_undiscounted, std_reward_undiscounted),
                      "episode_success={:.2f} +/- {:.2f}".format(mean_success, std_success),
                      "episode_collision={:.2f} +/- {:.2f}".format(mean_collision, std_collision))
                print("Episode length: {:.2f} +/- {:.2f}".format(mean_ep_length, std_ep_length))

            if mean_success < 1.0e-4 and self.n_calls % (self.eval_freq*5):
                self.model.setup_model()

            if mean_reward_discounted > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, 'best_model'))
                self.model.save(os.path.join(self.best_model_save_path, 'model' + str(self.num_timesteps)))
                self.best_mean_reward = mean_reward_discounted
                # Trigger callback if needed
                if self.callback is not None:
                    return self._on_event()

        return True

class WareHouse(gym.Env):
    def __init__(self, map, lot_max, input_max, arrvial_rate, instruction_rate, max_epi_length = None):
        self.map = map
        self.lot_max = lot_max
        self.input_max = input_max
        self.release_max = 20
        self.height, self.width = map.shape
        self.depot = None
        self.stations = []
        self.input_space = []
        self.max_epi_length = max_epi_length

        for c in range(self.width):
            for r in range(self.height):
                if map[r, c] == 'D':
                    self.depot = (r, c)
                elif map[r, c] == 'S':
                    self.stations.append((r, c))
                elif map[r, c] == 'I':
                    self.input_space = (r, c)

        # # near stations
        self.near0 = ((1, 0), (0, 1), (1, 2), (2, 1))
        self.near1 = ((3, 0), (4, 1), (2, 1), (3, 2))
        self.near2 = ((0, 3), (1, 4), (2, 3), (1, 2))
        self.near3 = ((3, 4), (4, 3), (3, 2), (2, 3))


        # # near input_space
        self.inear = ((4, 1), (3, 2), (4, 3))

        self.action_space = spaces.Discrete(5) # LEFT, UP, RIGHT, DOWN, STAY
        self.observation_space = spaces.MultiDiscrete([self.height, self.width, 2, self.input_max+1, self.release_max+1 ] + [self.lot_max+1]*(len(self.stations)*2) + [3, self.height, self.width, 2])
        # location x, y, load or not, input_amount, release, lot per stations

        self.arrvial_rate = arrvial_rate # per time unit
        self.instruction_rate = instruction_rate
        self.c_loss = 0.1 # loss cost per lot

        self.reward_per_success = 7.0
        self.reward_per_success2 = 10.0
        self.reward_per_success3 = 13.0
        self.reward_per_collision = 7.0
        self.reward_per_badaction = 3.0
        self.reward_per_blocked = 3.0

        self.viewer = None

        self.reset()

    def arrival(self):
        """
        :return number of newly arrived lots per station
        Assume poisson distribution with arrival rate
        """
        return np.random.poisson(self.arrvial_rate)

    def instruction(self):

        return np.random.poisson(self.instruction_rate)


    def get_cost(self, input_amount):
        """
        :return holding cost
        Compute cost given the current number of lots
        Assume coefficient*x^2
        """

        cost = 0.0
        coeff = 0.01
        x = input_amount
        cost += coeff*(x**2)
        return cost

    def get_cost2(self, release):

        cost2 = 0.0
        coeff2 = 0.005
        y = release
        cost2 += coeff2*(y**2)
        return cost2


    def step(self, action):
        row, col, load, input_amount, release, *lot_per_stations, sta, hrow, hcol, hstep = self.state
        prev_row, prev_col = row, col
        reward = 0
        if input_amount > 0 or lot_per_stations[4] > 0 or lot_per_stations[5] > 0 or lot_per_stations[6] > 0 or lot_per_stations[7] > 0:
            reward = -(self.get_cost(input_amount) + self.get_cost2(release))
        # print("Holding costs: ", reward)
        done = False
        self.steps += 1
        if self.steps == self.max_epi_length:
            done = True
        info = {}



        # New lot arrival
        arrv = self.arrival()

        inst = self.instruction()
        # print("New arrivals: ", arrv)
        input_amount = np.array(input_amount) + arrv

        release = np.array(release) + inst


        pc_loc = (prev_row, prev_col)
        h_loc = (hrow, hcol)

        #check(human)
        if h_loc in self.stations:
            if lot_per_stations[self.stations.index(h_loc)] > 0:
                lot_per_stations[self.stations.index(h_loc)] -= 1
                lot_per_stations[self.stations.index(h_loc) + 4] += 1

        if release > self.release_max:
            release = self.release_max



        #Move
        if action == 0: # LEFT
            col -= 1
        elif action == 1: # UP
            row -= 1
        elif action == 2: # RIGHT
            col += 1
        elif action == 3: # DOWN
            row += 1
        elif action == 4: # STAY
            row += 0
        else:
            raise Exception('bad action {}'.format(action))

        c_loc = (row, col)
        blocked = False



        #go input_space
        if pc_loc in self.inear:
            if input_amount > 0 and sta == 0 and load == 0 and lot_per_stations[5] == 0 and lot_per_stations[7] == 0:
                if c_loc != self.input_space:
                    reward -= self.reward_per_badaction
            elif release == 0 and sta == 0 and load == 0 and input_amount > 0 and lot_per_stations[5] >= 0 and lot_per_stations[7] >= 0:
                if c_loc != self.input_space:
                    reward -= self.reward_per_badaction


        # go station / bring lot
        bad_moves = [False, False, False, False]
        good_moves = [False, False, False, False]
        if pc_loc in self.near0:
            if lot_per_stations[0] + lot_per_stations[4] < self.lot_max and sta == 1 and load == 1:
                if c_loc not in self.stations[0]:
                    bad_moves[0] = True
                else:
                    good_moves[0] = True
        if pc_loc in self.near1:
            if lot_per_stations[1] + lot_per_stations[5] < self.lot_max and sta == 1 and load == 1:
                if c_loc not in self.stations[1]:
                    bad_moves[1] = True
                else:
                    good_moves[1] = True
        if pc_loc in self.near2:
            if lot_per_stations[2] + lot_per_stations[6] < self.lot_max and sta == 1 and load == 1:
                if c_loc not in self.stations[2]:
                    bad_moves[2] = True
                else:
                    good_moves[2] = True
        if pc_loc in self.near3:
            if lot_per_stations[3] + lot_per_stations[7] < self.lot_max and sta == 1 and load == 1:
                if c_loc not in self.stations[3]:
                    bad_moves[3] = True
                else:
                    good_moves[3] = True
        if sum(good_moves) == 0:
            reward -= self.reward_per_badaction*sum(bad_moves)

        bad_moves2 = [False, False, False, False]
        good_moves2 = [False, False, False, False]
        if pc_loc in self.near0:
            if lot_per_stations[4] > 0 and release > 0 and load == 0 and sta == 0:
                if c_loc not in (self.stations[0]):
                    bad_moves2[0] = True
                else:
                    good_moves2[0] = True
        if pc_loc in self.near1:
            if lot_per_stations[5] > 0 and release > 0 and load == 0 and sta == 0:
                if c_loc not in self.stations[1]:
                    bad_moves2[1] = True
                else:
                    good_moves2[1] = True
        if pc_loc in self.near2:
            if lot_per_stations[6] > 0 and release > 0 and load == 0 and sta == 0:
                if c_loc not in self.stations[2]:
                    bad_moves2[2] = True
                else:
                    good_moves2[2] = True
        if pc_loc in self.near3:
            if lot_per_stations[7] > 0 and release > 0 and load ==0 and sta == 0:
                if c_loc not in self.stations[3]:
                    bad_moves2[3] = True
                else:
                    good_moves2[3] = True

        if sum(good_moves2) == 0:
            reward -= self.reward_per_badaction*sum(bad_moves2)



        if col < 0 or col >= self.width or row < 0 or row >= self.height: # out of bounds, cannot move
            blocked = True
            reward -= self.reward_per_blocked


        elif c_loc in self.stations:
            if load == 0:
                if release > 0:
                    if lot_per_stations[self.stations.index(c_loc) + 4] > 0:
                        load = 1
                        sta = 2
                        lot_per_stations[self.stations.index(c_loc) + 4] -= 1
                        reward += self.reward_per_success
                elif release == 0:
                    if lot_per_stations[self.stations.index(c_loc) + 4] > 0:
                        load = 0

            elif load:
                if sta == 2:
                    load = 1
                elif sta == 1:
                    lot_per_stations[self.stations.index(c_loc)] += 1
                    if lot_per_stations[self.stations.index(c_loc)] + lot_per_stations[self.stations.index(c_loc) + 4] > self.lot_max:
                        load = 1
                        lot_per_stations[self.stations.index(c_loc)] -= 1
                    elif lot_per_stations[self.stations.index(c_loc)] + lot_per_stations[self.stations.index(c_loc) + 4] <= self.lot_max:
                        load = 0
                        sta = 0
                        reward += self.reward_per_success




        elif c_loc == self.depot: # into a goal cell
            if load:
                if sta == 2:
                    load = 0
                    sta = 0
                    release -= 1
                    reward += self.reward_per_success3
                    # print("Success move")
                    info['success'] = True
                elif release == 0:
                    load = 1


        elif c_loc == self.input_space:
            if input_amount > 0:
                if load:
                    blocked = True
                else:
                    load = 1
                    sta = 1
                    input_amount -= 1
                    reward += self.reward_per_success2



        # Set max lot_per_stations
        # for i in range(len(lot_per_stations)):
        #     if lot_per_stations[i] > self.lot_max:
        #         reward -= (lot_per_stations[i] - self.lot_max) * self.c_loss # Compute loss cost
        #         lot_per_stations[i] = self.lot_max

        # human move
        if lot_per_stations[0] > 0 and lot_per_stations[2] == 0:
            if hrow == 1 and hcol > 1 and hstep == 0:
                hstep += 1
            elif hrow == 1 and hcol > 1 and hstep == 1:
                hcol -= 1
                hstep -= 1
            elif hrow == 2 and hcol == 3 and hstep == 0:
                hstep += 1
            elif hrow == 2 and hcol == 3 and hstep == 1:
                if lot_per_stations[3] == 0:
                    hrow -= 1
                    hstep -= 1
                else:
                    if hrow == 1 and hcol == 1 and hstep == 0:
                        hstep += 1
                    elif hrow == 1 and hcol == 1 and hstep == 1:
                        hcol += 1
                        hstep -= 1
                    elif hrow == 1 and hcol == 2 and hstep == 0:
                        hstep += 1
                    elif hrow == 1 and hcol == 2 and hstep == 1:
                        hcol += 1
                        hstep -= 1
                    elif hrow == 1 and hcol == 3 and hstep == 0:
                        hstep += 1
                    elif hrow == 1 and hcol == 3 and hstep == 1:
                        hrow += 1
                        hstep -= 1
                    elif hrow == 2 and hcol == 3 and hstep == 0:
                        hstep += 1
                    elif hrow == 2 and hcol == 3 and hstep == 1:
                        hrow += 1
                        hstep -= 1
                    elif hrow == 3 and hcol == 3 and hstep == 0:
                        hstep += 1
                    elif hrow == 3 and hcol == 3 and hstep == 1:
                        hcol -= 1
                        hstep -= 1
                    elif hrow == 3 and hcol == 2 and hstep == 0:
                        hstep += 1
                    elif hrow == 3 and hcol == 2 and hstep == 1:
                        hcol -= 1
                        hstep -= 1
                    elif hrow == 3 and hcol == 1 and hstep == 0:
                        hstep += 1
                    elif hrow == 3 and hcol == 1 and hstep == 1:
                        hrow -= 1
                        hstep -= 1
                    elif hrow == 2 and hcol == 1 and hstep == 0:
                        hstep += 1
                    elif hrow == 2 and hcol == 1 and hstep == 1:
                        hrow -= 1
                        hstep -= 1
            else:
                if hrow == 1 and hcol == 1 and hstep == 0:
                    hstep += 1
                elif hrow == 1 and hcol == 1 and hstep == 1:
                    hcol += 1
                    hstep -= 1
                elif hrow == 1 and hcol == 2 and hstep == 0:
                    hstep += 1
                elif hrow == 1 and hcol == 2 and hstep == 1:
                    hcol += 1
                    hstep -= 1
                elif hrow == 1 and hcol == 3 and hstep == 0:
                    hstep += 1
                elif hrow == 1 and hcol == 3 and hstep == 1:
                    hrow += 1
                    hstep -= 1
                elif hrow == 2 and hcol == 3 and hstep == 0:
                    hstep += 1
                elif hrow == 2 and hcol == 3 and hstep == 1:
                    hrow += 1
                    hstep -= 1
                elif hrow == 3 and hcol == 3 and hstep == 0:
                    hstep += 1
                elif hrow == 3 and hcol == 3 and hstep == 1:
                    hcol -= 1
                    hstep -= 1
                elif hrow == 3 and hcol == 2 and hstep == 0:
                    hstep += 1
                elif hrow == 3 and hcol == 2 and hstep == 1:
                    hcol -= 1
                    hstep -= 1
                elif hrow == 3 and hcol == 1 and hstep == 0:
                    hstep += 1
                elif hrow == 3 and hcol == 1 and hstep == 1:
                    hrow -= 1
                    hstep -= 1
                elif hrow == 2 and hcol == 1 and hstep == 0:
                    hstep += 1
                elif hrow == 2 and hcol == 1 and hstep == 1:
                    hrow -= 1
                    hstep -= 1
        elif lot_per_stations[1] > 0 and lot_per_stations[0] == 0:
            if hrow == 1 and hcol == 1 and hstep == 0:
                hstep += 1
            elif hrow == 1 and hcol == 1 and hstep == 1:
                hrow += 1
                hstep -= 1
            elif hrow == 2 and hcol == 1 and hstep == 0:
                hstep += 1
            elif hrow == 2 and hcol == 1 and hstep == 1:
                hrow += 1
                hstep -= 1
            elif hrow == 1 and hcol == 2 and hstep == 0:
                hstep += 1
            elif hrow == 1 and hcol == 2 and hstep == 1:
                if lot_per_stations[2] == 0:
                    hcol -= 1
                    hstep -= 1
                else:
                    if hrow == 1 and hcol == 1 and hstep == 0:
                        hstep += 1
                    elif hrow == 1 and hcol == 1 and hstep == 1:
                        hcol += 1
                        hstep -= 1
                    elif hrow == 1 and hcol == 2 and hstep == 0:
                        hstep += 1
                    elif hrow == 1 and hcol == 2 and hstep == 1:
                        hcol += 1
                        hstep -= 1
                    elif hrow == 1 and hcol == 3 and hstep == 0:
                        hstep += 1
                    elif hrow == 1 and hcol == 3 and hstep == 1:
                        hrow += 1
                        hstep -= 1
                    elif hrow == 2 and hcol == 3 and hstep == 0:
                        hstep += 1
                    elif hrow == 2 and hcol == 3 and hstep == 1:
                        hrow += 1
                        hstep -= 1
                    elif hrow == 3 and hcol == 3 and hstep == 0:
                        hstep += 1
                    elif hrow == 3 and hcol == 3 and hstep == 1:
                        hcol -= 1
                        hstep -= 1
                    elif hrow == 3 and hcol == 2 and hstep == 0:
                        hstep += 1
                    elif hrow == 3 and hcol == 2 and hstep == 1:
                        hcol -= 1
                        hstep -= 1
                    elif hrow == 3 and hcol == 1 and hstep == 0:
                        hstep += 1
                    elif hrow == 3 and hcol == 1 and hstep == 1:
                        hrow -= 1
                        hstep -= 1
                    elif hrow == 2 and hcol == 1 and hstep == 0:
                        hstep += 1
                    elif hrow == 2 and hcol == 1 and hstep == 1:
                        hrow -= 1
                        hstep -= 1
            else:
                if hrow == 1 and hcol == 1 and hstep == 0:
                    hstep += 1
                elif hrow == 1 and hcol == 1 and hstep == 1:
                    hcol += 1
                    hstep -= 1
                elif hrow == 1 and hcol == 2 and hstep == 0:
                    hstep += 1
                elif hrow == 1 and hcol == 2 and hstep == 1:
                    hcol += 1
                    hstep -= 1
                elif hrow == 1 and hcol == 3 and hstep == 0:
                    hstep += 1
                elif hrow == 1 and hcol == 3 and hstep == 1:
                    hrow += 1
                    hstep -= 1
                elif hrow == 2 and hcol == 3 and hstep == 0:
                    hstep += 1
                elif hrow == 2 and hcol == 3 and hstep == 1:
                    hrow += 1
                    hstep -= 1
                elif hrow == 3 and hcol == 3 and hstep == 0:
                    hstep += 1
                elif hrow == 3 and hcol == 3 and hstep == 1:
                    hcol -= 1
                    hstep -= 1
                elif hrow == 3 and hcol == 2 and hstep == 0:
                    hstep += 1
                elif hrow == 3 and hcol == 2 and hstep == 1:
                    hcol -= 1
                    hstep -= 1
                elif hrow == 3 and hcol == 1 and hstep == 0:
                    hstep += 1
                elif hrow == 3 and hcol == 1 and hstep == 1:
                    hrow -= 1
                    hstep -= 1
                elif hrow == 2 and hcol == 1 and hstep == 0:
                    hstep += 1
                elif hrow == 2 and hcol == 1 and hstep == 1:
                    hrow -= 1
                    hstep -= 1
        elif lot_per_stations[2] > 0 and lot_per_stations[3] == 0:
            if hrow == 2 and hcol == 3 and hstep == 0:
                hstep += 1
            elif hrow == 2 and hcol == 3 and hstep == 1:
                hrow -= 1
                hstep -= 1
            elif hrow == 3 and hcol == 3 and hstep == 0:
                hstep += 1
            elif hrow == 3 and hcol == 3 and hstep == 1:
                hrow -= 1
                hstep -= 1
            elif hrow == 3 and hcol == 2 and hstep == 0:
                hstep += 1
            elif hrow == 3 and hcol == 2 and hstep == 1:
                if lot_per_stations[1] == 0:
                    hcol += 1
                    hstep -= 1
                else:
                    if hrow == 1 and hcol == 1 and hstep == 0:
                        hstep += 1
                    elif hrow == 1 and hcol == 1 and hstep == 1:
                        hcol += 1
                        hstep -= 1
                    elif hrow == 1 and hcol == 2 and hstep == 0:
                        hstep += 1
                    elif hrow == 1 and hcol == 2 and hstep == 1:
                        hcol += 1
                        hstep -= 1
                    elif hrow == 1 and hcol == 3 and hstep == 0:
                        hstep += 1
                    elif hrow == 1 and hcol == 3 and hstep == 1:
                        hrow += 1
                        hstep -= 1
                    elif hrow == 2 and hcol == 3 and hstep == 0:
                        hstep += 1
                    elif hrow == 2 and hcol == 3 and hstep == 1:
                        hrow += 1
                        hstep -= 1
                    elif hrow == 3 and hcol == 3 and hstep == 0:
                        hstep += 1
                    elif hrow == 3 and hcol == 3 and hstep == 1:
                        hcol -= 1
                        hstep -= 1
                    elif hrow == 3 and hcol == 2 and hstep == 0:
                        hstep += 1
                    elif hrow == 3 and hcol == 2 and hstep == 1:
                        hcol -= 1
                        hstep -= 1
                    elif hrow == 3 and hcol == 1 and hstep == 0:
                        hstep += 1
                    elif hrow == 3 and hcol == 1 and hstep == 1:
                        hrow -= 1
                        hstep -= 1
                    elif hrow == 2 and hcol == 1 and hstep == 0:
                        hstep += 1
                    elif hrow == 2 and hcol == 1 and hstep == 1:
                        hrow -= 1
                        hstep -= 1
            else:
                if hrow == 1 and hcol == 1 and hstep == 0:
                    hstep += 1
                elif hrow == 1 and hcol == 1 and hstep == 1:
                    hcol += 1
                    hstep -= 1
                elif hrow == 1 and hcol == 2 and hstep == 0:
                    hstep += 1
                elif hrow == 1 and hcol == 2 and hstep == 1:
                    hcol += 1
                    hstep -= 1
                elif hrow == 1 and hcol == 3 and hstep == 0:
                    hstep += 1
                elif hrow == 1 and hcol == 3 and hstep == 1:
                    hrow += 1
                    hstep -= 1
                elif hrow == 2 and hcol == 3 and hstep == 0:
                    hstep += 1
                elif hrow == 2 and hcol == 3 and hstep == 1:
                    hrow += 1
                    hstep -= 1
                elif hrow == 3 and hcol == 3 and hstep == 0:
                    hstep += 1
                elif hrow == 3 and hcol == 3 and hstep == 1:
                    hcol -= 1
                    hstep -= 1
                elif hrow == 3 and hcol == 2 and hstep == 0:
                    hstep += 1
                elif hrow == 3 and hcol == 2 and hstep == 1:
                    hcol -= 1
                    hstep -= 1
                elif hrow == 3 and hcol == 1 and hstep == 0:
                    hstep += 1
                elif hrow == 3 and hcol == 1 and hstep == 1:
                    hrow -= 1
                    hstep -= 1
                elif hrow == 2 and hcol == 1 and hstep == 0:
                    hstep += 1
                elif hrow == 2 and hcol == 1 and hstep == 1:
                    hrow -= 1
                    hstep -= 1
        elif lot_per_stations[3] > 0 and lot_per_stations[1] == 0:
            if hrow == 3 and hcol < 3 and hstep == 0:
                hstep += 1
            elif hrow == 3 and hcol < 3 and hstep == 1:
                hcol += 1
                hstep -= 1
            elif hrow == 2 and hcol == 1 and hstep == 0:
                hstep += 1
            elif hrow == 2 and hcol == 1 and hstep == 1:
                if lot_per_stations[0] == 0:
                    hrow += 1
                    hstep -= 1
                else:
                    if hrow == 1 and hcol == 1 and hstep == 0:
                        hstep += 1
                    elif hrow == 1 and hcol == 1 and hstep == 1:
                        hcol += 1
                        hstep -= 1
                    elif hrow == 1 and hcol == 2 and hstep == 0:
                        hstep += 1
                    elif hrow == 1 and hcol == 2 and hstep == 1:
                        hcol += 1
                        hstep -= 1
                    elif hrow == 1 and hcol == 3 and hstep == 0:
                        hstep += 1
                    elif hrow == 1 and hcol == 3 and hstep == 1:
                        hrow += 1
                        hstep -= 1
                    elif hrow == 2 and hcol == 3 and hstep == 0:
                        hstep += 1
                    elif hrow == 2 and hcol == 3 and hstep == 1:
                        hrow += 1
                        hstep -= 1
                    elif hrow == 3 and hcol == 3 and hstep == 0:
                        hstep += 1
                    elif hrow == 3 and hcol == 3 and hstep == 1:
                        hcol -= 1
                        hstep -= 1
                    elif hrow == 3 and hcol == 2 and hstep == 0:
                        hstep += 1
                    elif hrow == 3 and hcol == 2 and hstep == 1:
                        hcol -= 1
                        hstep -= 1
                    elif hrow == 3 and hcol == 1 and hstep == 0:
                        hstep += 1
                    elif hrow == 3 and hcol == 1 and hstep == 1:
                        hrow -= 1
                        hstep -= 1
                    elif hrow == 2 and hcol == 1 and hstep == 0:
                        hstep += 1
                    elif hrow == 2 and hcol == 1 and hstep == 1:
                        hrow -= 1
                        hstep -= 1
            else:
                if hrow == 1 and hcol == 1 and hstep == 0:
                    hstep += 1
                elif hrow == 1 and hcol == 1 and hstep == 1:
                    hcol += 1
                    hstep -= 1
                elif hrow == 1 and hcol == 2 and hstep == 0:
                    hstep += 1
                elif hrow == 1 and hcol == 2 and hstep == 1:
                    hcol += 1
                    hstep -= 1
                elif hrow == 1 and hcol == 3 and hstep == 0:
                    hstep += 1
                elif hrow == 1 and hcol == 3 and hstep == 1:
                    hrow += 1
                    hstep -= 1
                elif hrow == 2 and hcol == 3 and hstep == 0:
                    hstep += 1
                elif hrow == 2 and hcol == 3 and hstep == 1:
                    hrow += 1
                    hstep -= 1
                elif hrow == 3 and hcol == 3 and hstep == 0:
                    hstep += 1
                elif hrow == 3 and hcol == 3 and hstep == 1:
                    hcol -= 1
                    hstep -= 1
                elif hrow == 3 and hcol == 2 and hstep == 0:
                    hstep += 1
                elif hrow == 3 and hcol == 2 and hstep == 1:
                    hcol -= 1
                    hstep -= 1
                elif hrow == 3 and hcol == 1 and hstep == 0:
                    hstep += 1
                elif hrow == 3 and hcol == 1 and hstep == 1:
                    hrow -= 1
                    hstep -= 1
                elif hrow == 2 and hcol == 1 and hstep == 0:
                    hstep += 1
                elif hrow == 2 and hcol == 1 and hstep == 1:
                    hrow -= 1
                    hstep -= 1
        else:
            if hrow == 1 and hcol == 1 and hstep == 0:
                hstep += 1
            elif hrow == 1 and hcol == 1 and hstep == 1:
                hcol += 1
                hstep -= 1
            elif hrow == 1 and hcol == 2 and hstep == 0:
                hstep += 1
            elif hrow == 1 and hcol == 2 and hstep == 1:
                hcol += 1
                hstep -= 1
            elif hrow == 1 and hcol == 3 and hstep == 0:
                hstep += 1
            elif hrow == 1 and hcol == 3 and hstep == 1:
                hrow += 1
                hstep -= 1
            elif hrow == 2 and hcol == 3 and hstep == 0:
                hstep += 1
            elif hrow == 2 and hcol == 3 and hstep == 1:
                hrow += 1
                hstep -= 1
            elif hrow == 3 and hcol == 3 and hstep == 0:
                hstep += 1
            elif hrow == 3 and hcol == 3 and hstep == 1:
                hcol -= 1
                hstep -= 1
            elif hrow == 3 and hcol == 2 and hstep == 0:
                hstep += 1
            elif hrow == 3 and hcol == 2 and hstep == 1:
                hcol -= 1
                hstep -= 1
            elif hrow == 3 and hcol == 1 and hstep == 0:
                hstep += 1
            elif hrow == 3 and hcol == 1 and hstep == 1:
                hrow -= 1
                hstep -= 1
            elif hrow == 2 and hcol == 1 and hstep == 0:
                hstep += 1
            elif hrow == 2 and hcol == 1 and hstep == 1:
                hrow -= 1
                hstep -= 1

        # If human location = robot location ??
        if hrow == row and hcol == col:
            reward -= self.reward_per_collision
            info['collision'] = True




        if input_amount > self.input_max:
            reward -= (input_amount - self.input_max) * self.c_loss
            input_amount = self.input_max
        # lot_per_stations = [(self.lot_max if x > self.lot_max else x) for x in lot_per_stations]

        if blocked: # cannot move
            self.state = (prev_row, prev_col, load, input_amount, release) + tuple(lot_per_stations) + (sta, hrow, hcol, hstep)
            return np.array(self.state), reward, done, info
        else:
            self.state = c_loc + (load, input_amount, release) + tuple(lot_per_stations) + (sta, hrow, hcol, hstep)
            if sta == 1 and load == 1:
                reward += np.abs(prev_row-0) + np.abs(prev_col-2) - np.abs(c_loc[0]-0) - np.abs(c_loc[1]-2)
            return np.array(self.state), reward, done, info


    def reset(self):
        # Initial state is depot location without any lots at stations
        self.state = self.depot + (0, 0,) + (0,) * (1 + len(self.stations)*2) + (0,) + self.stations[0] + (0,)
        # self.state = np.array(self.depot + (0, 0,) + (0,) * (1 + len(self.stations)*2) + (0,) + self.stations[0] + (0,))
        # rand_state = self.observation_space.sample()
        # self.state[0:5] = rand_state[0:5]
        # self.state = self.observation_space.sample()
        self.steps = 0

        return np.array(self.state)  # reward, done, info can't be included

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        x_scale = screen_width/self.width
        y_scale = screen_height/self.height

        agent_radius = 0.2*min(x_scale, y_scale)
        human_radius = 0.1*min(x_scale, y_scale)
        lot_width = agent_radius
        station_color = (.5, .5, .5)
        lot_color = (0, 0, 0.8)
        check_color = (.9, .3, .0)
        input_color = (.6, .5, .3)

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            # Draw depot
            # self.depot = (r, c)
            l = self.depot[1] * x_scale
            r = (self.depot[1]+1) * x_scale
            t = screen_height - self.depot[0] * y_scale
            b = screen_height - (self.depot[0]+1) * y_scale
            depot = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            depot.set_color(.8, .8, 0)
            self.viewer.add_geom(depot)



            def get_lot():
                l, r, t, b = -lot_width / 2, lot_width / 2, lot_width / 2, -lot_width / 2
                return rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])

            self.geom_lots = [[] for a in range(len(self.stations))]
            self.geom_lotts = [[]]


            # Draw input_space
            l = self.input_space[1] * x_scale
            r = (self.input_space[1] + 1) * x_scale
            t = screen_height - self.input_space[0] * y_scale
            b = screen_height - (self.input_space[0] + 1) * y_scale
            input_space = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            input_space.set_color(input_color[0], input_color[1], input_color[2])
            self.viewer.add_geom(input_space)


            max_per_row = x_scale // (lot_width + lot_width*0.1)
            for i in range(self.input_max):
                lot = get_lot()
                lot.set_color(lot_color[0], lot_color[1], lot_color[2])
                lot.add_attr(rendering.Transform())

                r, c = divmod(i, max_per_row)
                lot_x = self.input_space[1] * x_scale +lot_width/2 + c * lot_width * 1.1
                lot_y = screen_height - self.input_space[0] * y_scale - lot_width/2 - r * lot_width *1.1

                lot.attrs[-1].set_translation(lot_x, lot_y)
                self.viewer.add_geom(lot)
                self.geom_lotts[0].append(lot)





            # Draw Station
            # self.stations.append((r, c))
            for idx, s in enumerate(self.stations):
                l = s[1] * x_scale
                r = (s[1] + 1) * x_scale
                t = screen_height - s[0] * y_scale
                b = screen_height - (s[0] + 1) * y_scale
                station = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
                station.set_color(station_color[0], station_color[1], station_color[2],)
                self.viewer.add_geom(station)


                for i in range(self.lot_max):
                    lot = get_lot()
                    lot.set_color(lot_color[0], lot_color[1], lot_color[2])
                    lot.add_attr(rendering.Transform())

                    r, c = divmod(i, max_per_row)
                    lot_x = s[1] * x_scale + lot_width/2 + c * lot_width * 1.1
                    lot_y = screen_height - s[0] * y_scale - lot_width/2 - r * lot_width * 1.1

                    lot.attrs[-1].set_translation(lot_x, lot_y)
                    self.viewer.add_geom(lot)
                    self.geom_lots[idx].append(lot)

            # Create grid
            xs = np.linspace(0, screen_width, self.width + 1)
            ys = np.linspace(0, screen_height, self.height + 1)
            for x in xs:
                xline = rendering.make_polyline([(x, 0), (x, screen_height)])
                xline.set_linewidth(1)
                self.viewer.add_geom(xline)
            for y in ys:
                yline = rendering.make_polyline([(0, y), (screen_width, y)])
                yline.set_linewidth(1)
                self.viewer.add_geom(yline)

            # Draw agent
            agent = rendering.make_circle(radius=agent_radius)
            self.agenttrans = rendering.Transform()
            agent.add_attr(self.agenttrans)
            self.viewer.add_geom(agent)

            # loaded pot
            # self.loaded = get_lot()
            # self.loadedtrans = rendering.Transform()
            # self.loaded.add_attr(self.loadedtrans)



            self.loaded = get_lot()
            self.loadedtrans = rendering.Transform()
            self.loaded.add_attr(self.loadedtrans)



            # Draw human
            human = rendering.make_circle(radius=human_radius)
            self.humantrans = rendering.Transform()
            human.add_attr(self.humantrans)
            human.set_color(.9,.6,.3)
            self.viewer.add_geom(human)

        x = self.state
        agentx = (x[1] + 0.5) * x_scale # MIDDLE OF AGENT
        agenty = screen_height - (x[0] + 0.5) * y_scale
        self.agenttrans.set_translation(agentx, agenty)
        if x[2] == 1 and x[13] == 1:  # loaded
            self.loaded.set_color(lot_color[0], lot_color[1], lot_color[2])
            self.loadedtrans.set_translation(agentx + agent_radius / 2, agenty - agent_radius / 2)
            self.viewer.add_onetime(self.loaded)
        elif x[2] == 1 and x[13] == 2:
            self.loaded.set_color(check_color[0], check_color[1], check_color[2])
            self.loadedtrans.set_translation(agentx + agent_radius / 2, agenty - agent_radius / 2)
            self.viewer.add_onetime(self.loaded)

        humanx = (x[15] + 0.5) * x_scale
        humany = screen_height - (x[14] + 0.5) * y_scale
        self.humantrans.set_translation(humanx, humany)



        for idx, s in enumerate(self.stations):
            for i in range(self.lot_max):
                if x[5 + idx] > 0 and x[9 + idx] == 0:
                    if i < x[5 + idx]:
                        self.geom_lots[idx][i].set_color(lot_color[0], lot_color[1], lot_color[2])
                    else:
                        self.geom_lots[idx][i].set_color(station_color[0], station_color[1], station_color[2])
                elif x[5 + idx] == 0 and x[9 + idx] > 0:
                    if i < x[9 + idx]:
                        self.geom_lots[idx][i].set_color(check_color[0], check_color[1], check_color[2])
                    else:
                        self.geom_lots[idx][i].set_color(station_color[0], station_color[1], station_color[2])
                elif x[9 + idx] > 0 and x[5 + idx] > 0:
                    if i < x[9 + idx]:
                        self.geom_lots[idx][i].set_color(check_color[0], check_color[1], check_color[2])
                        if i < x[5 + idx]:
                            self.geom_lots[idx][x[9 + idx] + i].set_color(lot_color[0], lot_color[1], lot_color[2])
                else:
                    self.geom_lots[idx][i].set_color(station_color[0], station_color[1], station_color[2])






        for i in range(self.input_max):
            if i < x[3]:
                self.geom_lotts[0][i].set_color(lot_color[0], lot_color[1], lot_color[2])
            else:
                self.geom_lotts[0][i].set_color(input_color[0], input_color[1], input_color[2])


        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

if __name__ == '__main__':
    MAP = [
        [' ', ' ', 'D', ' ', ' '],
        [' ', 'S', ' ', 'S', ' '],
        [' ', ' ', ' ', ' ', ' '],
        [' ', 'S', ' ', 'S', ' '],
        [' ', ' ', 'I', ' ', ' '],
    ]
    action_name = {0:"LEFT", 1:"UP", 2:"RIGHT", 3:"DOWN", 4:"STAY"}
    lot_max = 3
    input_max = 10
    arrvial_rate = 0.04
    instruction_rate = 0.04

    env = WareHouse(np.array(MAP),
                    lot_max=lot_max,
                    input_max=input_max,
                    arrvial_rate=arrvial_rate,
                    instruction_rate=instruction_rate,
                    max_epi_length=1440)
    #
    # Environment for evaluation
    evaluation_length = 1440 # evaluate for 1hour
    eval_env = WareHouse(np.array(MAP),
                         lot_max=lot_max,
                         input_max=input_max,
                         arrvial_rate=arrvial_rate,
                         instruction_rate=instruction_rate,
                         max_epi_length=evaluation_length)
    eval_env.seed(0)
    np.random.seed(0)
    cb = EvalCallback_Warehouse(eval_env=eval_env, n_eval_episodes=50, eval_freq=10000,
                                log_path="./model",
                                best_model_save_path="./best_model")
    model = PPO2('MlpPolicy', env, learning_rate = 1.0e-3, verbose=0) # 2.5e-2, 1.0e-2, 2.5e-3, 1.0e-3(제출본), 2.5e-4, 1.0e-4
    model.learn(total_timesteps=int(5e5), callback=cb)
    # # # # # #
    # #
    # # TEST
    # count_success = 0
    # cumul_reward = 0
    # env.reset()
    # for iter in range(10000):
    #     env.render()
    #     print("state: ", np.array(env.state))
    #     action = env.action_space.sample()
    #     observation, reward, _, info = env.step(action)
    #     cumul_reward += reward
    #     if info.get('success'):
    #         count_success += 1
    #     print("action: ", action_name[action])
    #     print("reward this step: ", reward)
    #     print("total reward: ", cumul_reward)
    #     print("="*50)
    #     time.sleep(1)
    # print("Total successful move: ", count_success)
    #
    #
    # # model = DQN('MlpPolicy', env, learning_rate=1e-3, prioritized_replay=True, verbose=1)
    # model = PPO2('MlpPolicy', env, verbose=1)
    # # Train the agent
    # model.learn(total_timesteps=int(1e5))
    # Save the agent
    # model.save("dqn_lunar")
    # del model  # delete trained model to demonstrate loading

    # Load the trained agent
    # model = DQN.load("dqn_lunar")
    #
    # Evaluate the agent
    eval_env.seed(0)
    np.random.seed(0)
    eval_env = DummyVecEnv([lambda: eval_env])
    epi_rewards_undiscounted, epi_rewards_discounted, epi_success_moves, epi_collisions, episode_lengths \
        = evaluate_policy_Warehouse(model, eval_env,
                                    n_eval_episodes=50,
                                    render=False,
                                    deterministic=True,
                                    return_episode_rewards=True)

    mean_reward_undiscounted, std_reward_undiscounted = np.mean(epi_rewards_undiscounted), np.std(
        epi_rewards_undiscounted)
    mean_reward_discounted, std_reward_discounted = np.mean(epi_rewards_discounted), np.std(epi_rewards_discounted)
    mean_success, std_success = np.mean(epi_success_moves), np.std(epi_success_moves)
    mean_collision, std_collision = np.mean(epi_collisions), np.std(epi_collisions)
    mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)

    print(
          "episode_discounted_reward={:.2f} +/- {:.2f}".format(mean_reward_discounted,
                                                               std_reward_discounted),
          "episode_undiscounted_reward={:.2f} +/- {:.2f}".format(mean_reward_undiscounted, std_reward_undiscounted),
          "episode_success={:.2f} +/- {:.2f}".format(mean_success, std_success),
          "episode_collision={:.2f} +/- {:.2f}".format(mean_collision, std_collision))
    print("Episode length: {:.2f} +/- {:.2f}".format(mean_ep_length, std_ep_length))
    # # mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=50)

    # # # Enjoy trained agent
    # model = PPO2.load("C://Users//admin//Desktop//best_model//best_model.zip")
    # render = True
    # obs = env.reset()
    # count_success = 0
    # cumul_reward = 0
    # for i in range(1000):
    #     env.render()
    #     print("state: ", np.array(env.state))
    #     action, _states = model.predict(obs)
    #     obs, rewards, dones, info = env.step(action)
    #
    #     cumul_reward += rewards
    #     if info.get('success'):
    #         count_success += 1
    #     print("action: ", action_name[action])
    #     print("reward this step: ", rewards)
    #     print("total reward: ", cumul_reward)
    #     print("="*50)
    #     if render:
    #         time.sleep(0.5)
    #
    # print("Total successful move: ", count_success)