from mss import mss
import pydirectinput
import cv2
import numpy as np
import pytesseract
from matplotlib import pyplot as plt
import time
from gym import Env
from gym.spaces import Box, Discrete
import os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import env_checker
from stable_baselines3 import DQN
import re


class WebGame(Env):
    def __init__(self):
        super().__init__()

        self.observation_space = Box(low=0, high=255, shape=(1, 100, 100), dtype=np.uint8)
        self.action_space = Discrete(3)

        self.cap = mss()
        self.game_location = {'top': 730, 'left': 180, 'width': 500, 'height': 260}
        self.done_location = {'top': 590, 'left': 850, 'width': 875, 'height': 100}
        self.score_location = {'top': 460, 'left': 2260, 'width': 270, 'height': 80}

    def step(self, action):
        global x
        action_map = {
            0: 'space',
            1: 'down',
            2: 'no_op'
        }

        if action != 2:
            pydirectinput.press(action_map[action])


        done, done_cap = self.get_done()
        new_observation = self.get_observation()
        info = {}

        if done:
            time.sleep(0.2)
            try:
                reward = int(self.get_score())
                if reward < 50:
                    reward = 5
                elif reward == 606:
                    reward = 66
                print(reward)
            except ValueError:
                reward = 50
                print("Value error!")
                print(reward)
        else:
            reward = 1

        return new_observation, reward, done, info

    def render(self):
        cv2.imshow('Game', np.array(self.cap.grab(self.game_location))[:, :, :3])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.close()

    def reset(self):
        time.sleep(1)
        pydirectinput.click(x=250, y=250)
        pydirectinput.press('space')
        global x
        x = 0

        return self.get_observation()

    def get_observation(self):
        raw = np.array(self.cap.grab(self.game_location))[:, :, :3].astype(np.uint8)
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (100, 100))
        channel = np.reshape(resized, (1, 100, 100))

        return channel

    def get_done(self):
        done_cap = np.array(self.cap.grab(self.done_location))[:, :, :3]
        done_strings = ['GAME', 'GAHE']
        done = False
        res = pytesseract.image_to_string(done_cap)[:4]

        if res in done_strings:
            done = True

        return done, done_cap

    def get_score(self):
        score_cap = np.array(self.cap.grab(self.score_location))[:, :, :3]
        res = pytesseract.image_to_string(score_cap)
        res = re.sub(r"[\n\t\s]*", "", res)
        res = res.replace('O', '0')
        res = res.replace('o', '0')
        res = res.lstrip('0')

        return res


class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
env = WebGame()
# env_checker.check_env(env)
CHECKPOINT_DIR = 'train'
LOG_DIR = 'logs'
callback = TrainAndLoggingCallback(check_freq=1000, save_path=CHECKPOINT_DIR)
model = DQN('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, buffer_size=1800000, learning_starts=100, learning_rate=0.0001)
# model = model.load(os.path.join('train', 'best_model_5000'))
model.learn(total_timesteps=300000, callback=callback)


"""
plt.imshow(cv2.cvtColor(env.get_observation()[0], cv2.COLOR_GRAY2RGB))
plt.show()
"""

"""

for episode in range(10):
    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs)
        print(env.step(int(action)))
        obs, reward, done, info = env.step(int(action))
        total_reward += reward
    print(f'Total reward for episode {episode} is {total_reward}!')
    
"""

"""
env.render()
# plt.imshow(cv2.cvtColor(env.get_observation()[0], cv2.COLOR_GRAY2RGB))
done, done_cap = env.get_done()
print(done)
plt.imshow(np.array(done_cap))
while True:
    env.render()
    time.sleep(1)
plt.show()

"""
