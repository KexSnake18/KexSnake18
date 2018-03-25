# Inspired by https://keon.io/deep-q-learning/
import pickle

import os
import random
import gym
import gym_ple
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import keras
import json

from DQN.RewardFunction import RewardFunction
from keras.models import model_from_json
from collections import deque
from keras.models import Sequential
from keras.optimizers import Adam

from builtins import str

class DQNSolver:
    
    def __init__(self, env_name="Snake-v0", max_mem=100000, target_update_freq=10, observetime=500, n_episodes=1000, n_win_ticks=195, 
                 max_env_steps=None, gamma=0.9, epsilon=1.0, epsilon_min=0.1, 
                 epsilon_log_decay=0.995, alpha=0.01, alpha_decay=0.01, batch_size=32, monitor=False, quiet=False, 
                 reward_func=RewardFunction()):
        self.memory = deque(maxlen=max_mem)
        self.env = gym.make(env_name)
        self.env_name = env_name
        if monitor: self.env = gym.wrappers.Monitor(self.env, '../data/cartpole-1', force=True)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_log_decay
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.n_episodes = n_episodes
        self.observetime = observetime
        self.n_win_ticks = n_win_ticks
        self.batch_size = batch_size
        self.quiet = quiet
        if max_env_steps is not None: self.env._max_episode_steps = max_env_steps
        
        self.model = None
        self.target_update_freq = target_update_freq
        self.target_model = None
        self.reward_func = reward_func
        self.episode_runs = 0

    def save(self, agent_path):
        model = self.model
        target_model = self.target_model
        env = self.env
        self.model = None # can't be unpickled
        self.target_model = None # can't be unpickled
        self.env = None # can't be unpickled
        
        with open(agent_path, 'wb') as f:
            pickle.dump(self, f, 0)
            
        self.model = model
        self.target_model = target_model
        self.env = env

    @staticmethod
    def load(agent_path):
        with open(agent_path, 'rb') as f:
            agent = pickle.load(f)
            agent.env = gym.make(agent.env_name)
            return agent
        
    def summary(self):
        print("Episodes:", self.n_episodes)
        print("Episode runs:", self.episode_runs)
        print("Exploring rate:", self.epsilon)
        print("Reward discount:", self.gamma)
        print("Memory length:", len(self.memory))
        print("Memory max length:", self.memory.maxlen)
        
    def _confirm_model_load(self):
        if self.model:
            raise Exception("Model allready loaded")
            #return DQNUtils.yes_no_question("Model already loaded, create new?", no_print="No model created...")
        return True
        
    def save_model(self, path):
        #self.model.save(os.path.abspath(os.path.join(path, "{0}${1}".format(agent_name, model_name))))
        self.model.save(path)

    def load_model(self, model):
        """model is a Sequential or a path to the .h5 file of the saved model.
        """
        if self._confirm_model_load():
            if isinstance(model, Sequential):
                self.model = model
            elif isinstance(model, str):
                ext = os.path.splitext(model)[-1]
                if ext == ".h5":
                    self.model = keras.models.load_model(model)
                    
                elif ext == ".json":
                    with open(model) as f:
                        self.model = model_from_json(json.load(f))
                        self.model.compile(loss='mse', optimizer=Adam(lr=self.alpha, decay=self.alpha_decay))
                else:
                    raise Exception("Loading model '{}' failed".format(model))
                
            if self.target_update_freq:
                self.target_model = keras.models.clone_model(self.model)
                self.target_model.set_weights(self.model.get_weights())
    
    def action_to_str_snake(self, action):
        actions = ["UP", "LEFT", "RIGHT", "DOWN", "NO_ACTION"]
        return actions[action]

    def get_output(self, state):
        return DQNModel.get_layer_outputs(self.model, state)
        
    def predict(self, state):
        """Returns action (int)
        """
        pred = self.model.predict(state)
        #print(pred[0])
        return np.argmax(pred)
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state, epsilon):
        """Choose action by prediction or random with prabability epsilon
        """
        return self.env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(self.model.predict(state))

    def get_epsilon(self, t):
        #return max(self.epsilon_min, min(self.epsilon, 1.0 - math.log10((t + 1) * self.epsilon_decay)))
        return max(self.epsilon_min, self.epsilon)

    def target_q(self, next_state):
        if self.target_model:
            return np.max(self.target_model.predict(next_state)[0])
        else:
            return np.max(self.model.predict(next_state)[0])
                      
    def preprocess_state(self, state):
        """Scale pixels and reshape
        """
        state[...] = state[...]/255 
        return state.reshape((1,) + self.model.input_shape[1:])
            
    def play(self, render=False, n_games=1):
        """Plays specified number of games and print the accumulated reward
        """
        all_rewards = []
        while True:
            state = self.preprocess_state(self.env.reset())
            done = False
            i = 0
            tot_reward = 0
            while not done:
                if render:
                    self.env.render()
                    time.sleep(0.15)
                action = self.predict(state)
                state, reward, done, info = self.env.step(action)
                state = self.preprocess_state(state)
                tot_reward += self.reward_func(1, reward)
                #print(reward)
                i += 1
                if i % 500 == 0:
                    #if not input("continue 500 ticks?") == "y":
                    print("Too many ticks")
                    break
            all_rewards.append(tot_reward)
            #print("Play reward:", tot_reward)
            n_games -= 1
            if n_games <= 0:
                return all_rewards
        
    def play_from_memory(self, index):
        """Show all states as pictures from a game sequence in memory
        """
        tot_reward = 0
        done = False
        while not done:
            state, action, reward, next_state, done = self.memory[index]
            tot_reward += reward
            plt.imshow(state.reshape(64, 64, 3))
            plt.show()
            input()
            index += 1
        print("Total reward:", tot_reward)
    
    def replay(self, batch_size):
        """Fit the model with batches from the memory
        """
        x_batch, y_batch = [], []
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, done in minibatch:
            y_target = self.model.predict(state)
            #y_target[0][action] = reward if done else reward + self.gamma * np.max(self.model.predict(next_state)[0])
            y_target[0][action] = reward if done else reward + self.gamma * self.target_q(next_state)
            x_batch.append(state[0])
            y_batch.append(y_target[0])
        
        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min
        
    def explore_replay(self, episode, replay=True, render=False, remember=False, observetime=500, break_on_done=True):
        """Trial and error playing which will be stored in memory
        """
        state = self.preprocess_state(self.env.reset())
        done = False
        #i = 0
        for t in range(observetime):
            if render:
                time.sleep(0.01)
                self.env.render()
            action = self.choose_action(state, self.get_epsilon(episode))
            next_state, reward, done, _ = self.env.step(action)
            reward = self.reward_func(episode, reward)
            next_state = self.preprocess_state(next_state)
            if remember: self.remember(state, action, reward, next_state, done)
            state = next_state
            
            #if replay: self.replay(self.batch_size)
            #i += 1
            if done:
                if break_on_done:
                    break
                state = self.preprocess_state(self.env.reset())
                done = False
                #i = 0
        if render:
            self.env.close()
        
    def run(self, n_episodes=None, play=True):
        """Runs training of the agent with the specified number of episodes
        """
        if n_episodes is not None: 
            self.n_episodes = n_episodes
        all_rewards = deque(maxlen=100)
        
        # Initialize memory with 5000 observations
        if not len(self.memory) >= 1000:
            print("Initializing memory...")
            self.explore_replay(0, replay=False, observetime=1000, break_on_done=False)
            print("Initialize complete")
        else:
            print("Initialization not needed")
        
        for e in range(1, self.n_episodes+1):
            self.episode_runs += 1
            
            self.explore_replay(e, remember=True, observetime=self.observetime)
            self.replay(self.batch_size)###
            # Update target network
            if self.target_model and (self.episode_runs % self.target_update_freq == 0):
                self.target_model.set_weights(self.model.get_weights())
                print("Updated target network")
            
            if play: 
                rewards = self.play() # Play once every episode to check performance
                all_rewards.extend(rewards)
            mean_score = np.mean(all_rewards)
            
            if e % 10 == 0 and not self.quiet:
                print('[Episode {}] - Mean accumulated reward over last 100 episodes was {}.'.format(self.episode_runs, mean_score))
            if e % 100 == 0 and not self.quiet:    
                print("Best game score from last 100 games:", np.max(all_rewards))
            
        if not self.quiet: print('Did not solve after {} episodes 😞'.format(e))
        return all_rewards

    def user_play(self):
        """User choose action with input
        """
        state = self.preprocess_state(self.env.reset())
        done = False
        i = 0
        while True:
            self.env.render()
            action = input("Action: ")
            if action == "q":
                print("Quitting game...")
                break
            self.env.step(int(action))
            print(self.action_to_str_snake(int(action)))
        self.env.close()

if __name__ == '__main__':
    #agent.save_model("Snaker-v2")
    #print(pickle.dump.__doc__)
    #agent = DQNSolver("Snake-v0", batch_size=32)
    #agent.load_model("Snaker_v1")
    #agent.save("test_agent")
    #agent.play(n_games=5, render=True)
    #agent.env.close()
    
    agent_name = "TestAgent"
    model_name = "Snaker_v1"
    agent = DQNSolver(n_episodes=1)
    agent.load_model(model_name)
    agent.run(n_episodes=1)
    #agent = load_agent("test_agent", model_name)
    agent.save(agent_name, model_name)
    
    
    