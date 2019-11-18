from ple import PLE
import pickle
import frogger_new
import numpy as np
import numpy as array
import pygame.constants
#from pygame.constants import K_w,K_a,K_F15,K_s,K_d
import copy
import frog_sprites
from itertools import product
import csv
import random
import time
import sys
from sklearn.preprocessing import normalize

states = []
obss = []
feature_size = 4 # should increase it when adding more features
died = 0 
won = 0
qtable= [float(0) for col in range(feature_size)]
w = np.random.uniform(-1,1,feature_size)
class NaiveAgent():
    state ={'home', 'win', 'midway', 'downmid', 'death', 'start'}
    qtable = [float(0) for col in range(6)]
    def __init__(self, actions):
        self.actions = actions
        self.step = 0
        self.NOOP = pygame.constants.K_F15
        self.current_state= 0 # baghie ham az jense rewardan vali ba index neshune gozari mishan 
        self.gamma = 0.9        #   

    def pickAction(self, game, obs, previous_states, w):
        act = pygame.constants.K_F15
        q_s_left = 0
        q_s_right =0
        q_s_up = 0
        q_s_down = 0
        action_id=0
        q_s_a =0
        frg_x = obs['frog_x']
        frg_y = obs['frog_y']
        frog_w = obs['rect_w']
        frog_h = obs['rect_h']
        feture_of_state = [float(0) for col in range(feature_size)]
        current_q_s = [float(0) for col in range(5)]
        f_x_s = [float(0) for col in range(feature_size)]
        temp_s = [float(0) for col in range(feature_size)] # to keep Q(s,a) (since we are not going to save the table of Q it is a variable only)
        temp_s_NOOP = [float(0) for col in range(feature_size)]
        temp_s_left=[float(0) for col in range(feature_size)]
        temp_s_right=[float(0) for col in range(feature_size)]
        temp_s_up=[float(0) for col in range(feature_size)]
        temp_s_down=[float(0) for col in range(feature_size)]
        # do nothing or stay action
        temp_s_NOOP = state_features(game, obs,previous_states,frg_x,frg_y, frog_w,frog_h)
        #print(temp_s_NOOP)
        for i in range(feature_size-1):
            current_q_s[3] += w[i]*temp_s_NOOP[i]
        # observe left 
        frg_x = move_left(obs['frog_x'], frog_w)
        #get new state represntation 
        temp_s_left =state_features(game, obs, previous_states, frg_x,obs['frog_y'], frog_w,frog_h)
        for i in range(feature_size-1):
            current_q_s[1] += w[i]*temp_s_left[i]
        # observe right
        frg_x= move_right(obs['frog_x'], frog_w)
        temp_s_right =state_features(game, obs, previous_states ,frg_x,obs['frog_y'],frog_w,frog_h)
        for i in range(feature_size):
            current_q_s[2]+=  w[i]*temp_s_right[i]
        # observe up
        frg_y= move_up(obs['frog_y'],frog_h)
        temp_s_up =state_features(game, obs , previous_states,obs['frog_x'],frg_y,frog_w,frog_h)
        #print("UUUp", temp_s_up)
        for i in range(feature_size):
            current_q_s[0]+= w[i]*temp_s_up[i]
        #move down
        frg_y = move_down(obs['frog_y'],frog_h)
        temp_s_down =state_features(game, obs, previous_states, obs['frog_x'],frg_y,frog_w,frog_h)
        for i in range(feature_size):
            #q_s_down + = w[i]*temp_s[i]
            current_q_s[4]+=  w[i]*temp_s_down[i]
        # which one is the best action now also return 
        action_id=np.argmax(current_q_s) # return this 
        q_s_a = current_q_s[action_id] # return this value
        #feture_of_state = f_x_s[action_id]
        if action_id == 0:
            act = pygame.constants.K_w # up
            feture_of_state = temp_s_up

        elif action_id ==2: #right
            act = pygame.constants.K_d 
            feture_of_state = temp_s_right

        elif action_id ==1: #left
            act =  pygame.constants.K_a 
            feture_of_state = temp_s_left

        elif action_id ==4 :#down
            act = pygame.constants.K_s
            feture_of_state = temp_s_down
        elif action_id ==3: # staye
            act = pygame.constants.K_F15
            feture_of_state = temp_s_NOOP 

        if(random.uniform(0, 1)>0.95):
            nrand = random.uniform(0,1)
            if(nrand<0.25):
                act = pygame.constants.K_w
            elif(nrand<0.5):
                act = pygame.constants.K_a
            elif(nrand<0.75):
                act = pygame.constants.K_s
            else:
                act = pygame.constants.K_d
            
        #print(feture_of_state)
        #time.sleep(.3)
        return [act,q_s_a, feture_of_state]#
        #return self.NOOP , q_s_a, feture_of_state
       # return self.actions[np.random.randint(0,len(self.actions))]
#def move to left
def move_left(frg_x,frog_w):
    xx = frg_x - frog_w
    return xx
def move_right(frg_x,frog_w):
    xx = frg_x+frog_w
    return xx
def move_up(frg_y, frog_h):
    yy = frg_y-frog_h
    return yy
def move_down(frg_y,frog_h):
    yy = frg_y+frog_h
    return yy

def state_features(game,obs, previous_states, frg_x,frg_y,frog_w,frog_h):
    # boundray the frog can see I will change it later , MAYBE
    boundray = 5
    boundray2 = 2
    x_left = max(0, frg_x-boundray*frog_w) 
    x_right = min(frg_x+boundray*frog_w, 768)
    y_top = min(frg_y-boundray*frog_h,0)
    y_bottom = max(512,frg_y+boundray*frog_h) 
    direction = [0 for i in range(len(obs['cars']))] # we need direction of each cars
    direction_riv = [0 for i in range(len(obs['rivers']))]# we need direction of each river
    #speed =[]
    riv_count=0
    countcars=0
    safe_counters =0
    not_safe_counters =0
    count_obstacles = 0
    not_safe_overall=0
    positive_overall=0
    right_danger = 0 #right side dangerous
    left_danger =0 #left side danger
    up_danger = 0
    down_danger = 0
    #is_dangerous = [float(0) for col in range(len(obs['cars']))]
    #is_car_or_not = [float(0) for col in range(len(obs['cars']))]
    is_car=-1
    is_dangerous_car =-1
    for i in range(len(obs['cars'])):
        #print(obs['cars'][i])   
        # calculate the coordinates of each cars
        car_i_width = obs['cars'][i].w
        car_i_hight = obs['cars'][i].h
        car_i_left = obs['cars'][i].left
        car_i_right = obs['cars'][i].right
        car_i_top = obs['cars'][i].top
        car_i_butom = obs['cars'][i].bottom
        car_i_x = obs['cars'][i].x
        car_i_y = obs['cars'][i].y
        #print [x_left, y_top, x_right, y_bottom],[car_i_left, car_i_top, car_i_right, car_i_butom]
        if( bb_intersection_over_union([x_left, y_top, x_right, y_bottom],[car_i_left, car_i_top, car_i_right, car_i_butom])):
            countcars=countcars+1
            is_car = 1
            dis = frg_x - obs['cars'][i].x 
            disy= frg_y - obs['cars'][i].y
            dirct = (previous_states['cars'][i].x)-(obs['cars'][i].x)
            count_obstacles = count_obstacles +1
            #print(dirct)
        #print(previous_states['cars'])
        # to check whether they are comming toward the frog or geeting fars
        # dis<0 frog <car
        # dir<0 
            if (dis <=0 and dirct<0) or (dis >= 0 and dirct >0):
                safe_counters = safe_counters +1
                #print ("this car is getting far",safe_counters)
            elif (dis<=0  and dirct >0) or (dis > 0 and dirct <0):
                not_safe_counters = not_safe_counters+1
                #not_safe_overall = not_safe_overall+1
                is_dangerous_car=1
            #print dis,disy,frg_x,frg_y,obs['cars'][i].x,obs['cars'][i].y
            if(abs(dis)<=boundray2*frog_w and abs(disy)<frog_h/2 and dis<0): #right side (we used frog_h /2 to make sure we two objects are in the same row with a litle tolerance)
                right_danger += 1
            if(abs(dis)<=boundray2*frog_w and abs(disy)<0.5*frog_h and dis>0): #left side
                left_danger += 1
            if(abs(dis)<frog_w and abs(disy)<2*frog_h and abs(disy)>frog_h/2 and disy>0): #if it is near frog_x and between 0.5 and 2 of frog height in y-cordinate, it is above the frog
                up_danger += 1
            if(abs(dis)<frog_w and abs(disy)<2*frog_h and abs(disy)>frog_h/2 and disy<0): #similar for down but with -1*heigh
                down_danger += 1
     
    count_turtle =0  
    count_bridge = 0
    # to count turtules and logs
    is_river_or_not = [float(0) for col in range(len(obs['rivers']))]
    rivers_f = [float(0) for col in range(len(obs['rivers']))]
    is_bridge = -1
    is_turtle=-1
    count_stars=0
    river_right_danger = 0
    river_left_danger = 0
    river_up_danger = 0
    river_down_danger = 0
    mid_river_up_danger = 0
    for i in range(len(obs['rivers'])):
        #print(obs['rivers'][i])   
        # calculate the coordinates of each objects (turtles, logs) 
        riv_i_width = obs['rivers'][i].w
        riv_i_hight = obs['rivers'][i].h
        riv_i_left = obs['rivers'][i].left
        riv_i_right = obs['rivers'][i].right
        riv_i_top = obs['rivers'][i].top
        riv_i_butom = obs['rivers'][i].bottom
        riv_i_x = obs['rivers'][i].x
        riv_i_y = obs['rivers'][i].y
        #unaafected if, check every single river
        if( True or bb_intersection_over_union([x_left, y_top, x_right, y_bottom],[riv_i_left, riv_i_top, riv_i_right, riv_i_butom])):
            #print 'herh'
            riv_count=riv_count+1
            #is_river_or_not[i] =1
            # count number of bridge (those which have the height of 20)
            #in brdiges and turtules we should not only consider rivers.x since it may be longer than one tutrle or a long tree
            #by considering only rivers.x we may loos keep track of that river
            #instead we should consider the nearest value of that brige. for that matter we consider start and end of an brige as 
            # a bounding and find the nearest part and consider dis for that
            x1 =obs['rivers'][i].x  #x values of starting point of brdige to frog
            x2 =  (obs['rivers'][i].x  + obs['rivers'][i].w) #x values of ending point of brdige to frog
            all_x = range(int(x1-1),int(x2+1))#all possible distances 
            
            #dis = frg_x - obs['rivers'][i].x 
            list_dis = [x - y for x, y in zip(([frg_x]*len(all_x)),all_x)]#a list containing all possible distance from frog to river i
            #find index of minimum of abs. we use abs to make sure the size of distance matteres. for example btwn dist of -3 0 3, zero is 
            # is the min dist, but without abs we will get -3 as min
            min_ind = np.argmin(map(abs, list_dis))
            dis = float(list_dis[min_ind])
            #time.sleep(0.1)
            disy = frg_y - obs['rivers'][i].y 
            count_bridge = count_bridge +1
            #is_bridge[i]=1
            is_bridge =1
            # direction of this bridge, but how to use this direction
            dirct_bridge = (previous_states['rivers'][i].x)-(obs['rivers'][i].x)

            #same as cars for brige
        
            #if the river has a dis<2 and disy<2 means, this is the river we are in

            #in that case if (frg_x+frog_w > river[i].x+river[i].w) == True means
            #frog can not goes right
            if( ((frg_x+frog_w-2)>(obs['rivers'][i].x+obs['rivers'][i].w) ) and (dis)<2 and abs(disy)<2):
                river_right_danger += 1
            #if (frg_x-1 < river[i].x == True means frog can not goes left
            if(frg_x-1<obs['rivers'][i].x and abs(dis)<frog_w and abs(disy)<2):
                river_left_danger += 1
            
            #if (xDistance <frogw and frog_w<ydistance<2*frog_w and disy>0) then turtles and bridges are on the top
            if(abs(dis)<2 and abs(disy)>=frog_h-3 and abs(disy)<(2*frog_h)-3 and disy>0): 
                river_up_danger += 1
            #when we are in the middle way, recognizing turtules are a little bit tricky, so we define a
            #especial if for that line (y==229) to see if we are below a turtle or not
            if(obs['rivers'][i].y==229 and frg_x>=obs['rivers'][i].left and frg_x<=obs['rivers'][i].right):
                mid_river_up_danger = 1
            
            #with the same condition but disy<0, turtles and bridges are in the bottom
            if(abs(dis)<2 and abs(disy)>=frog_h-3 and abs(disy)<(2*frog_h)-3 and disy<0): 
                river_down_danger += 1

    #we need to make sure if the frog is in the river side of the game or car side of the game
    river_side = False 
    upper_river = False # to make sure it is not on the middle line
    for i in range(len(obs['rivers'])):
        #check if the frog is near any reiver by y-coordinate or not
        if(abs(obs['rivers'][i].y - frg_y)<2*frog_h):
            river_side = True
        if(obs['rivers'][i].y >= frg_y): #frog is upper than at least one river
            river_side = True
            upper_river = True
            #check for left or right danger as the frog is on a bridge/turtle and it is getting out of the game
            if(frg_x<0):
                river_left_danger = 1 
            if(frg_x>425): #possible right danger
                river_right_danger = 1
            break; #it is river side, do not to explore for more


    #since we count turtles and brdiges as cars we need to reverse the danger
    #for exampel if left_danger = 1 that means there is a turtle or brige in left, so left is safe
    #then we swithc 1 by 0

    #we only count turtle and bridges for "UP" and "Down" so it means there is no danger
    #we need to revers it to make sure if it is danger or not
    if(river_side): #we are in river side  part, then change river_danger with main dangers
        #print river_up_danger,'river_up_danger'
        if(river_up_danger>0): 
            up_danger = 0
        else: 
            up_danger = 1
        if(upper_river): #for down we must be aboce the the middle line
            if(river_down_danger>0): down_danger = 0
            else: down_danger = 1
        if(not(upper_river) and mid_river_up_danger==0):
            up_danger = 1
    
    #if the frog is on the first row of turtles, that means behind turtles we have ony empy field
    #so othere is no danger at all. But since we don't count anything behind frog and later we switch
    #0 to 1, it seems there is a danger. We devised this if to make sure we encounter danger correctly
    if(frg_y==229): #no down danger, since we are on the first row of rivers
        down_danger = 0

    #we can't non-river places for left and right, so there are real dangers
    if(upper_river):
        right_danger = river_right_danger
        left_danger = river_left_danger

    #check if top is the home, in this case no danger for the top
    home_above = 1 #there is no home above at first
    for i in range(len(obs['homeR'])):
        if(frg_y==101): #near the home vertically
            #there is a home and the home doesn't have any crocodile
            if(frg_x>=obs['homeR'][i].left-3 and frg_x+frog_w<=obs['homeR'][i].right+3):
                #time.sleep(1)
                #exit(0)
                #if(frg_x==obs['frog_x']):
                    #print "vaisa"
                if((obs['homes'][i]==0.33)):#crocidle is here
                    home_above = 0
                    break #there is a danger, no need to check others
                #else:
                #    home_above = 1 #it is safe
                #    break #if we find one, that means there is a place to go, no need to check others
                
    
    #if there is no home above, so there is a up_danger
    if(home_above<1 and frg_y<110 and frg_y>90):
        up_danger = 1
    elif(home_above>0 and frg_y<110 and frg_y>90):
        up_danger = 0

    #is empty spot around the frog
    f_set = [up_danger,right_danger,left_danger,down_danger]
    return f_set

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    if interArea > 0:
        return True
    else: 
        return False

game = frogger_new.Frogger()
fps = 30
p = PLE(game, fps=fps,force_fps=False) # environment object..
agent = NaiveAgent(p.getActionSet())
global qtable 
#global w
global feature_size

#global feture_of_state # features of state S produced by pickAction function
#initialize weights randomly
w = np.random.uniform(-1,1,feature_size)
if(len(sys.argv)==2):
#loading weights file based on command argument
    with open(str(sys.argv[1])) as f:  # Python 3: open(..., 'rb')
        w = pickle.load(f)
obs = game.getGameState()
current = obs
while True:
    gamma = 0.9
    alpha = 0.1 # consider it low 
    current_q_prime_s = [float(0) for col in range(5)]
    current_q_s = [float(0) for col in range(5)]
    if p.game_over():
        p.reset_game()
    game.step(0)
    old_frg_y = obs['frog_y']
    #get the state here 
    #update the Q function before choosing the next actions
    previous_states=current 
    obs = game.getGameState()
    current = game.getGameState()
    action, q_s_a, fetures_previous = agent.pickAction(game, obs, previous_states,w) # update Q table and give me the best actiion
    # so that you can calculate q_s'_a' for all action and get max
    # find the max value of that next state
    reward = p.act(action)
    if reward >= float(1.0): # later save it in a file and read it from hear
        gold_weights = w
        print "number of times frog died: ",died," -- number of times frog won: ",won
        won += 1
        time.sleep(10)
        #save each win weight separately
        with open('weights'+str(won) +'.pkl', 'w') as f:  # Python 3: open(..., 'wb')
            pickle.dump(gold_weights, f)
        #exit()
    #get the feature of next state 
    pre_for_next = current
    obs = game.getGameState()
    #print(obs)
    q_s_prime_a = [0,0,0,0,0]
    frg_x = obs['frog_x']
    frg_y = obs['frog_y']
    frog_w = obs['rect_w']
    frog_h = obs['rect_h']
    # do nothing or stay
    temp_s = state_features(game, obs,pre_for_next, frg_x,frg_y, frog_w,frog_h)
    if reward == float(-1.0): # later save it in a file and read it from hear
        died +=1
        print "number of times frog died: ",died," -- number of times frog won: ",won
        states.append(temp_s)
        obss.extend(obs)
    #print(temp_s)
    for i in range(feature_size):
        current_q_prime_s[4] += w[i]*temp_s[i]
    #observe left 
    frg_x_orig = frg_x
    frg_y_orig = frg_y
    frg_x = move_left(frg_x_orig, frog_w)
    #get new state represntation 
    temp_s = state_features(game, obs,pre_for_next, frg_x,obs['frog_y'], frog_w,frog_h)
    #q_s_left = w[0]*temp_s[0] + w[1]*temp_s[1]
    for i in range(feature_size):
        #q_s_left + = w[i]*temp_s[i]
        current_q_prime_s[0] += w[i]*temp_s[i]
    # observe right
    frg_x= move_right(frg_x_orig, frog_w)
    temp_s = state_features(game, obs,pre_for_next, frg_x,obs['frog_y'],frog_w,frog_h)
    for i in range(feature_size):
        #q_s_right + = w[i]*temp_s[i]
        current_q_prime_s[1]+=  w[i]*temp_s[i]
    # observe up
    frg_y= move_up(frg_y_orig,frog_h)
    temp_s = state_features(game, obs, pre_for_next,obs['frog_x'],frg_y,frog_w,frog_h)
    for i in range(feature_size):
        #q_s_up + = w[i]*temp_s[i]
        current_q_prime_s[2]+= w[i]*temp_s[i]
    #move down
    frg_y = move_down(frg_y_orig,frog_h)
    temp_s = state_features(game, obs , pre_for_next,obs['frog_x'],frg_y,frog_w,frog_h)
    for i in range(feature_size):
        #q_s_down + = w[i]*temp_s[i]
        current_q_prime_s[3]+=  w[i]*temp_s[i]
    q_sprime_max = max(current_q_prime_s)
    q_sample = reward + gamma * q_sprime_max     #q_sample = reward + max_of_next_state
    # observe the sorrending then get the Q(s', a') calculate max_of_next_state # again need to get the surrounding 
    for i in range(feature_size):
        w[i] = w[i]+alpha*(q_sample - q_s_a)*fetures_previous[i]
    w = w / np.linalg.norm(w)