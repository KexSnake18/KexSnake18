import os
import sys
import keras
import numpy as np
import json
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from DQN.DQNSolver import DQNSolver
from DQN.RewardFunction import RewardFunction
from keras.models import model_from_json
from pyexpat import model
from collections import deque
            
class Trainer:
    def __init__(self):
        self.path_models = os.path.abspath(os.path.relpath("Templates/Models/"))
        self.path_agents = os.path.abspath(os.path.relpath("Templates/Agents/"))
        self.path_trained = os.path.abspath(os.path.relpath("Trained/"))
        self.path_random = os.path.abspath(os.path.relpath("Trained/RandomModels"))
        
    def load_agent_template(self, agent_name):
        return DQNSolver.load(os.path.abspath(os.path.join(self.path_agents, agent_name)))
    
    def load_model_template(self, model_name):
        """Return uncomplied model
        """
        path = os.path.abspath(os.path.join(self.path_models, model_name + ".json"))
        with open(path) as f:
            model = model_from_json(json.load(f))
        return model
        
    def trained_name(self, agent_name, model_name):
        return agent_name + "$" + model_name
        
    def clean_templates(self):
        for x in os.listdir(self.path_models):
            os.remove(os.path.join(self.path_models, x))
        for x in os.listdir(self.path_agents):
            os.remove(os.path.join(self.path_agents, x))
        
    def clean_trained(self):
        for x in os.listdir(self.path_trained):
            os.remove(os.path.join(self.path_trained, x))        
     
    def check_agent(self, agent_name):
        agent = DQNSolver.load(os.path.abspath(os.path.join(self.path_agents, agent_name)))
        agent.summary()
     
    def check_agent_status(self, agent_name, model_name):
        agent = DQNSolver.load(os.path.abspath(os.path.join(self.path_trained, self.trained_name(agent_name, model_name), agent_name)))
        agent.load_model(os.path.abspath(os.path.join(self.path_trained, self.trained_name(agent_name, model_name), model_name + ".h5")))
        #agent.save(os.path.abspath(os.path.join(self.path_trained, self.trained_name(agent_name, model_name), agent_name)))
        agent.summary()
        
    def play(self, agent_name, model_name, n_games=1):
        tr_name = self.trained_name(agent_name, model_name)
        agent = DQNSolver.load(os.path.abspath(os.path.join(self.path_trained, tr_name, agent_name)))
        agent.load_model(os.path.abspath(os.path.join(self.path_trained, tr_name, model_name + ".h5")))
        agent.summary()###
        agent.play(render=True, n_games=n_games)
        agent.env.close()
        
    def resume(self, agent_name, model_name, n_episodes=100):
        tr_name = self.trained_name(agent_name, model_name)
        agent = DQNSolver.load(os.path.abspath(os.path.join(self.path_trained, tr_name, agent_name)))
        agent.load_model(os.path.abspath(os.path.join(self.path_trained, tr_name, model_name + ".h5")))
        print("Resuming {} ...".format(tr_name))
        agent.run(n_episodes=n_episodes)
        agent.save(os.path.abspath(os.path.join(self.path_trained, tr_name, agent_name)))
        agent.save_model(os.path.join(self.path_trained, tr_name, model_name + ".h5"))
        
    def resume_random_model(self, agent_name, model_name, n_episodes=100):
        """Loads the specified agent template and the model from RandomModels.
        Will only save the model that is being resumed, not the agent.
        """
        agent = self.load_agent_template(agent_name)
        agent.load_model(os.path.join(self.path_random, model_name + ".h5"))
        print("Resuming random model {} ...".format(model_name + ".h5"))
        agent.run(n_episodes=n_episodes)
        agent.save_model(os.path.join(self.path_random, model_name + ".h5"))
        
    def train(self, agent_name, model_name, overwrite=False):
        """
        """
        tr_name = self.trained_name(agent_name, model_name)
        save_dir = os.path.join(self.path_trained, tr_name)
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        save_path_a = os.path.join(save_dir, agent_name)
        save_path_m = os.path.join(save_dir, model_name + ".h5")
        if (os.path.isfile(save_path_a) or os.path.isfile(save_path_m)) and overwrite is False:
            print("'{}' allready exists, no new model trained.".format(tr_name))
            return
        agent = DQNSolver.load(os.path.abspath(os.path.join(self.path_agents, agent_name)))
        agent.load_model(os.path.abspath(os.path.join(self.path_models, model_name + ".json")))
        agent.model.summary()###
        agent.run()
        agent.save(save_path_a)
        print("Agent '{}' created".format(agent_name))
        agent.save_model(save_path_m)
        print("Model '{}' created".format(model_name))
        
    def __run(self, agent_name, model_name, overwrite=False):
        """Not yet implemented
        """
        for agent_name in os.listdir(self.path_agents):
            save_dir = os.path.join(self.path_trained, agent_name)
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
            for model_template in os.listdir(self.path_models):
                model_name = os.path.splitext(model_template)[0]
                if model_name in skip_models:
                    continue
                model_name = "{0}${1}.h5".format(agent_name, model_name)
                save_path = os.path.join(save_dir, model_name)
                if os.path.isfile(save_path) and overwrite is False:
                    print("Model '{}' allready exists, no new model trained.".format(model_name))
                    continue
                agent = DQNSolver.load(os.path.abspath(os.path.join(self.path_agents, agent_name)))
                agent.load_model(os.path.abspath(os.path.join(self.path_models, model_template)))
                agent.model.summary()###
                agent.run()
                agent.save_model(save_path)
                print("Model '{}' created".format(model_name))
                
    def test_stuff(self):
        agent_name = "Agent_2"
        model_name = "Atari"
        n_games = 5
        agent = DQNSolver.load(os.path.abspath(os.path.join(self.path_agents, agent_name)))
        #agent.save(os.path.abspath(os.path.join(self.path_trained, self.trained_name(agent_name, model_name), agent_name)))
        #agent.load_model(os.path.abspath(os.path.join(self.path_random, model_name + ".h5")))
        agent.summary()###
        #agent.play(render=True, n_games=n_games)
        #agent.env.close()
            
if __name__ == "__main__":
    
    #np.random.seed(123)
    agent = "Agent_2"
    model = "Atari"
    trainer = Trainer()
    
    #trainer.test_stuff()
    
    #trainer.train(agent, model, overwrite=True) # Train agent/model from templates
    #trainer.resume(agent, model, n_episodes=1000) # Resume training of agent/model
    #trainer.resume_random_model(agent, model, n_episodes=10000) # Resume training of "random" model with agent template
    trainer.play(agent, model, n_games=10) # Play with trained agent/model
    
    #trainer.check_agent(agent)
    #trainer.check_agent_status(agent, model) # Check parameters of trained agent
    #trainer.add_agents() # Add agents to templates
    #trainer.clean_templates() # Clean all agent/model templates
    #trainer.clean_trained() # Clean all trained agents/models (doesn't work)