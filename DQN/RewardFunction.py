class RewardFunction:
    def __init__(self, update_times=[1], reward_mapping=[None]):
        self.update_times = update_times # List
        self.reward_mapping = reward_mapping # Dict
        
    def __call__(self, episode, reward):
        #for t in update_times:
        #    if episode >= t:
        #        reward = self.reward_mapping(t-1)
        # Just for snake
        if reward == 0:
            reward = -1
        elif reward == 1:
            reward = 100
        elif reward == -5:
            reward = -100
        return reward
    
if __name__ == "__main__":
    r = RewardFunction()
    for i in range(10):
        print(r(1, i)) 
        