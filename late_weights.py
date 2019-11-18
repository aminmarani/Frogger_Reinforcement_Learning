from ple import PLE
import frogger_new
import numpy as np
from pygame.constants import K_w,K_a,K_F15

class NaiveAgent():
    def __init__(self, actions):
        self.actions = actions
        self.step = 0
        self.NOOP = K_F15

    def pickAction(self, reward, obs):
        return self.NOOP
        #Uncomment the following line to get random actions
        #return self.actions[np.random.randint(0,len(self.actions))]

game = frogger_new.Frogger()
fps = 30
p = PLE(game, fps=fps,force_fps=False)
agent = NaiveAgent(p.getActionSet())
reward = 0.0

#p.init()

while True:
    if p.game_over():
        p.reset_game()

    obs = game.getGameState()
    #print obs
    action = agent.pickAction(reward, obs)
    reward = p.act(action)
    #print game.score
    print("frog location",obs['frog_x'],obs['frog_y'])
    print("REward",reward) 
    if reward == float(1.0): # later save it in a file and read it from hear
        gold_weights = w
    elif reward == float(+2.0):
        print("it has reached the middle way")
# I guess remove this part
        # print("TEMP SSssssssssssSSS", temp_s)
        # if (len(AllStates) == 0):
        #    AllStates.append(temp_s)
        #    print("Statesss",AllStates)
        # else:
        #     for ss in range(len(AllStates)-1):
        #         if temp_s == AllStates[ss]:
        #             key_state = ss # to check its Q value 
        #             max_q_value_sprime = np.max(qtable[ss])
        #             print("REPETETIVE ONE", max_q_value_sprime)
        #         else:
        #             print("NEW ONE")
        #             AllStates.append(temp_s) 
        #             max_q_value_sprime = 0 # it has never this state before so it is zero, but not need to append it 
        #             # how to save id of each states
        #             #qtable.append()
        #     print("qtable",qtable)    
        #     # riversi do it as for cars