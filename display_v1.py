import gym
import pandas as pd
from gym import spaces
import numpy as np
from gym.utils import seeding # random seed control 위해 import

import os
from stable_baselines.common.vec_env import VecEnv, sync_envs_normalization, DummyVecEnv
from typing import Union, List, Optional, Tuple

from stable_baselines import DQN, PPO2
from stable_baselines.common.callbacks import EvalCallback, BaseCallback

class Displapy(gym.Env):
    def __init__(self, order_data, setup_time, device_max_1, device_max_2):
        self.order = order_data
        self.setup_time = setup_time


        ##model production time
        self.p_m1_time = 40
        self.p_m2_time = 70
        self.p_m3_time = 50
        self.p_m4_time = 40


        ##model change time
        #same device
        self.change_m1_m2 = 6
        self.change_m2_m1 = 6
        self.change_m3_m4 = 6
        self.change_m4_m3 = 6

        #different device
        self.change_m1_m3 = 13
        self.change_m1_m4 = 13
        self.change_m2_m3 = 13
        self.change_m2_m4 = 13
        self.change_m3_m1 = 13
        self.change_m3_m2 = 13
        self.change_m4_m1 = 13
        self.change_m4_m2 = 13


        self.p_m_time = max(self.p_m1_time, self.p_m2_time, self.p_m3_time, self.p_m4_time)

        self.device_max = max(device_max_1, device_max_2)
        self.device_type = 0, 1


        self.action_space = spaces.Discrete(3) # SETUP1, SETUP2, STAY
        self.observation_space = spaces.MultiDiscrete([self.p_m_time+1, self.device_max, self.device_type, 12, 31, 24])

        self.c_loss = 0.5 # loss cost per lot





        # When 1 model is completed
        self.reward_per_success = 3

        # Meet maximum uptime
        self.reward_operation_max = 10.0

        # Penalty for low utilization
        self.reward_operation_rate1 = 5

        # Penalty for low utilization
        self.reward_operation_rate2 = 2.5

        self.viewer = None

        self.reset()


    def excess_penalty(self, day, due_date):
        penalty1 = 0
        coef1 = 0.01
        x = abs(due_date[2] - day)
        penalty1 += coef1*(x**2)

        return penalty1

    def abandon_penalty(self, p_amount_set_up):
        penalty2 = 0
        coef2 = 0.3
        y = p_amount_set_up
        penalty2 += coef2 * (y**2)

        return penalty2

    def step(self, action):
        required_time, amount_set_up, device, month, day, time = self.state
        p_amount_set_up = amount_set_up
        required_5 = 0
        set_5 = 0
        model_5 = 0



        # 여기에 정보 하나씩 불러오는 코드 구성할 것 (order 파일에서 납기순, 주문량순으로 불러옴)
        required_info = self.order[(self.order != 0).any(axis = 1)].iloc[self.steps]
        required_amount = required_info[]

        # consider defective
        if required_info[0] == BLK_12:
            required_time = ((required_amount / 0.  975) / 506) * 40
        elif required_info[0] == BLK_2:
            required_time = ((required_amount / 0.975) / 506) * 70
        elif required_info[0] == BLK_3:
            required_time = ((required_amount / 0.975) / 400) * 50
        elif required_info[0] == BLK_4:
            required_time = ((required_amount / 0.975) / 400) * 40


        reward = 0

        if amount_set_up != p_amount_set_up:
            reward -= self.abandon_penalty

        # 납기를 지키지 못한 것에 대한 코드 구성할 것
        if self.order[time][0] == month and self.order[time][1] < day:
            reward -= self.excess_penalty
        elif month - self.order[time][0] == 1:
            reward -= self.excess_penalty
        elif month - self.order[time][0] > 1:
            (month - self.order[time][0]) * 30


        #penalty for low utilization
        utilization = required_5 / required_5 + set_5 + model_5
        if self.steps % 5 == 0:
            if utilization < 0.7:
                reward -= self.reward_operation_rate1
            elif utilization < 0.75:
                reward -= self.reward_operation_rate2





        done = False
        self.steps += 1
        info = {}


        #set-up action
        if action == 0:
            amount_set_up = device_max_1
            device = 0
            set_5 += 1
        elif action == 1:
            amount_set_up = device_max_2
            device = 1
            set_5 += 1
        elif action == 2:
            amount_set_up += 0
            device += 0
        else:
            raise Exception('bad action {}'.format(action))


        if required_time >= amount_set_up:
            required_5 += required_time
            required_time == 0
        elif required_time == amount_set_up:

        if amount_set_up == 0:
            reward += self.reward_operation_max


        self.state = (required_time, amount_set_up, device, month, day, time)
        return np.array(self.state), reward, done, info

    def reset(self):
        self.state = (0, 0, 0,) + (4, 1, 0,)

        self.steps = 0

if __name__ == '__main__':
    action_name = {0: "Device 1 set-up", 1: "Device 2 set-up", 2: "Not set-up"}

    order_data = pd.read_csv('order.csv')
    setup_time = 28
    device_max_1 = 150
    device_max_2 = 130
    env = Displapy(order_data=order_data, setup_time=setup_time, device_max_1=device_max_1, device_max_2=device_max_2)

