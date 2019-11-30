# Reusing Maze environment

import numpy as np
import matplotlib.pyplot as plt

import time
import os
import random
from collections import defaultdict

from IPython import display
from PIL import Image
from mpl_toolkits.axes_grid1 import ImageGrid
from tqdm import tqdm 

# Some colours
LIGHT_RED = '#FFC4CC'
LIGHT_GREEN = '#95FD99'
BLACK = '#000000'
WHITE = '#FFFFFF'
LIGHT_PURPLE = '#E8D0FF'
LIGHT_ORANGE = '#FAE0C3'


class Town:

    # Actions
    STAY = 0
    MOVE_LEFT = 1
    MOVE_RIGHT = 2
    MOVE_UP = 3
    MOVE_DOWN = 4

    # Give names to actions
    actions_names = {
        STAY: "stay",
        MOVE_LEFT: "move left",
        MOVE_RIGHT: "move right",
        MOVE_UP: "move up",
        MOVE_DOWN: "move down"
    }

    # Reward values
    STEP_REWARD = 0
    IMPOSSIBLE_REWARD = -100
    ROB_REWARD = 1
    CAUGHT_REWARD = -10

    def __init__(self, maze, weights=None, random_rewards=False, police_cant_stay=True):
        """ Constructor of the environment Maze.
        """
        self.maze = maze
        self.minotaur_cant_stay = police_cant_stay
        self.actions = self.__actions()
        self.states, self.map = self.__states()
        self.n_actions = len(self.actions)
        self.n_states = len(self.states)
        self.transition_probabilities = self.__transitions()
        self.rewards = self.__rewards(weights=weights,
                                      random_rewards=random_rewards)

    def __actions(self):
        actions = dict()
        actions[self.STAY] = (0, 0)
        actions[self.MOVE_LEFT] = (0, -1)
        actions[self.MOVE_RIGHT] = (0, 1)
        actions[self.MOVE_UP] = (-1, 0)
        actions[self.MOVE_DOWN] = (1, 0)
        return actions

    def __states(self):
        states = dict()
        map = dict()
        end = False
        s = 0
        # Player position
        for pi in range(self.maze.shape[0]):
            for pj in range(self.maze.shape[1]):
                # Minotaur position
                for mi in range(self.maze.shape[0]):
                    for mj in range(self.maze.shape[1]):
                        # All combinations of player and minotaur
                        # inside the maze and player not in wall
                        if self.maze[pi, pj] != 1:
                            states[s] = (pi, pj, mi, mj)
                            map[(pi, pj, mi, mj)] = s
                            s += 1
        # NO win or dead like in minotaur
        return states, map

    def __move(self, state, action, for_transition_prob=False):
        """ Player makes a step in the maze, given a current position and an action.
            If the action STAY or an inadmissible action is used, the player stays in place.
            Simultaneously the minotaur makes the move
            
            for_transition_prob --
                returns the len(l) of valid minotaur positions to set t_prob to 1/l

            :return tuple next_state: 
                (Px,Py,Mx,My) on the maze that player and minotaur transitions to.
        """
        # For the player
        # Compute the future position given current (state, action)

        row = self.states[state][0] + self.actions[action][0]
        col = self.states[state][1] + self.actions[action][1]
        # For minotaur 
        # Play a random valid action
        valid_minotaur_moves = self.__minotaur_actions(state, cant_stay=self.minotaur_cant_stay)
        minotaur_pos = random.choice(valid_minotaur_moves)
        # Is the future position an impossible one ?
        hitting_maze_walls = (row == -1) or (row == self.maze.shape[0]) or \
                             (col == -1) or (col == self.maze.shape[1]) or \
                             (self.maze[row, col] == 1)
        # Based on the impossiblity check return the next state.
        
        if for_transition_prob:
            # We can let minotaur take its turn but
            # we would have to handle that in rewards to check if action results in hitting wall 
            # Instead of checking if action != stay and position of player remains the same,
            # we could make minotaur also stay so we can just see if the state has changed
            # its just simpler
            if hitting_maze_walls:
                # same state
                return  self.states[state][0], self.states[state][1], \
                        [[self.states[state][2], self.states[state][3]]]
            else: 
                return row, col, valid_minotaur_moves
        if hitting_maze_walls:
            return state
        else:
            return self.map[(row, col, minotaur_pos[0], minotaur_pos[1])]

    def __minotaur_actions(self, state, cant_stay=True):
        # Random action for minotaur
        pos = (self.states[state][2], self.states[state][3])
        valid_moves = []
        # Get all valid actions for the minotaur position
        valid_actions = list(self.actions.keys())
        if cant_stay and self.STAY in valid_actions:
            valid_actions.remove(self.STAY)
        for action in valid_actions:
            row = pos[0] + self.actions[action][0]
            col = pos[1] + self.actions[action][1]
            outside_maze = (row == -1) or (row == self.maze.shape[0]) or \
                (col == -1) or (col == self.maze.shape[1])
            # Assuming minotaur can stay/walk within the walls
            if not outside_maze:
                valid_moves.append([row, col])

        return valid_moves

    def __transitions(self):
        """ Computes the transition probabilities for every state action pair.
            :return numpy.tensor transition probabilities: tensor of transition
            probabilities of dimension S*S*A
        """
        # Initialize the transition probailities tensor (S,S,A)
        dimensions = (self.n_states, self.n_states, self.n_actions)
        transition_probabilities = np.zeros(dimensions)

        # Compute the transition probabilities. 
        # Note that the transitions are probabilistic based on minotaur's random move
        for s in range(self.n_states):
            for a in range(self.n_actions):
                row, col, valid_minotaur_moves = self.__move(s, a, for_transition_prob=True)
                for minotaur_pos in valid_minotaur_moves:
                    next_s = self.map[(row, col, minotaur_pos[0], minotaur_pos[1])]
                    transition_probabilities[next_s, s, a] = 1/len(valid_minotaur_moves)
        return transition_probabilities

    def __rewards(self, weights=None, random_rewards=None):

        rewards = np.zeros((self.n_states, self.n_actions))

        # If the rewards are not described by a weight matrix
        if weights is None:
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    next_s = self.__move(s, a)
                    # Reward for hitting a wall
                    if s == next_s and a != self.STAY:
                        rewards[s, a] = self.IMPOSSIBLE_REWARD
                    # Reward for being in the terminal state

                    # Check if the player position was at 2/win position while taking the action a
                    elif self.is_win(s):
                        rewards[s, a] = self.ROB_REWARD
                    # Reward for landing in the same cell as minotaur i.e being DEAD
                    # Check if the player position is equal to that of minotaur while taking the action a
                    elif self.is_dead(s):
                        rewards[s, a] = self.CAUGHT_REWARD
                    # Reward for taking a step to an empty cell that is not the exit
                    else:
                        rewards[s, a] = self.STEP_REWARD

        # If the weights are described by a weight matrix
        else:
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    next_s = self.__move(s, a)
                    i, j = self.states[next_s]
                    # Simply put the reward as the weights o the next state.
                    rewards[s, a] = weights[i][j]

        return rewards

    def is_win(self, s):
        return (self.maze[self.states[s][0:2]] == 2) and not self.is_dead(s)  

    def is_dead(self, s):
        return self.states[s][0:2] == self.states[s][2:4]

    def show(self):
        print('The states are :')
        print(self.states)
        print('The actions are:')
        print(self.actions)
        print('The mapping of the states:')
        print(self.map)
        print('The rewards:')
        print(self.rewards)
    
    def get_next_state(self, state, action):
        return self.__move(state, action)


def QLearning(env, start, log_states, steps = 10000000):
    """ Solves the shortest path problem using dynamic programming
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input int horizon        : The time T up to which we solve the problem.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """

    # The dynamic prgramming requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    r = env.rewards
    n_states = env.n_states
    states = env.states
    n_actions = env.n_actions
    actions = env.actions
    map = env.map
    lambd = 0.8
    
    Q = np.zeros((n_states, n_actions))
    # number of updates of Q[s, a]
    n = np.zeros((n_states, n_actions))
    start_state = map[start]
    s = start_state
    values_dic = defaultdict(list)
    for _ in tqdm(range(steps)):
        a = random.choice(list(actions.keys()))
        next_s = env.get_next_state(s, a)
        alpha = 1/pow(n[s, a]+1, 2/3)
        Q[s, a] += alpha * (r[s, a] + lambd * max(Q[next_s]) - Q[s, a])
        for state in log_states:
            values_dic[str(state)].append(np.max(Q[state]))
        n[s, a] += 1
        s = next_s
    _ = plt.title("Q-Learning")
    _ = plt.xlabel("Number of Steps")
    _ = plt.ylabel("Value Function")

    for x in values_dic.keys():
        _ = plt.plot(values_dic[x], label=states[int(x)])
    _ = plt.legend(loc=0)
    _ = plt.savefig("./Q-learning.png")
    
    return Q

def SARSA(env, start, log_states, epsilon=0.1, steps = 10000000):
    """ Solves the shortest path problem using dynamic programming
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input int horizon        : The time T up to which we solve the problem.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """

    # The dynamic prgramming requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    r = env.rewards
    n_states = env.n_states
    states = env.states
    n_actions = env.n_actions
    actions = env.actions
    map = env.map
    lambd = 0.8
    epsilon = epsilon
    Q = np.zeros((n_states, n_actions))
    # number of updates of Q[s, a]
    n = np.zeros((n_states, n_actions))
    start_state = map[start]
    s = start_state

    # Epsilon greedy    
    max_a = np.argmax(Q[s])
    action_probs = np.dot([1]*n_actions, epsilon/n_actions)
    action_probs[max_a] += 1 - epsilon
    a = np.random.choice(list(actions.keys()), p=action_probs)
    
    values_dic = defaultdict(list)
    for _ in tqdm(range(steps)):

        next_s = env.get_next_state(s, a)
        alpha = 1/pow(n[s, a]+1, 2/3)

        # Epsilon greedy
        max_a = np.argmax(Q[s])
        action_probs = np.dot([1]*n_actions, epsilon/n_actions)
        action_probs[max_a] += 1 - epsilon
        next_a = np.random.choice(list(actions.keys()), p=action_probs)
        Q[s, a] += alpha * (r[s, a] + lambd * Q[next_s][next_a] - Q[s, a])

        for state in log_states:
            values_dic[str(state)].append(np.max(Q[state]))
        
        n[s, a] += 1
        s = next_s
        a = next_a

    _ = plt.title(f"SARSA Epsilon {str(epsilon)}")
    _ = plt.xlabel("Number of Steps")
    _ = plt.ylabel("Value Function")

    for x in values_dic.keys():
        _ = plt.plot(values_dic[x], label=states[int(x)])
    _ = plt.legend(loc=0)
    _ = plt.savefig(f"./SARSA_{str(epsilon)}.png")
    
    return Q

def draw_town(maze):
    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK,
               2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED}

    # Give a color to each cell
    rows, cols = maze.shape
    colored_maze = [[col_map[maze[j, i]]
                     for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows))

    # Remove the axis ticks and add title title
    ax = plt.gca()
    ax.set_title('The Town')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    rows, cols = maze.shape
    colored_maze = [[col_map[maze[j, i]]
                     for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_maze,
                     cellLoc='center',
                     loc=(0, 0),
                     edges='closed')
    # Modify the hight and width of the cells in the table
    tc = grid.properties()['child_artists']
    for cell in tc:
        cell.set_height(1.0/rows)
        cell.set_width(1.0/cols)


if __name__ == '__main__':
    
    # Description of the maze as a numpy array
    town = np.array([
        [0, 0, 0, 0],
        [0, 2, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]    
    ])

    env = Town(town, police_cant_stay=False)
    start  = (0,0,3,3)
    # States of interest
    SOI = [start,(1,1,3,3),(0,0,1,1),(3,3,0,0)]
    log_states = []
    for x in SOI:
        log_states.append(env.map[x])

    Q = SARSA(env, start, log_states, epsilon=0.1)
    # # Simulate the shortest path starting from position A
    # method = 'DynProg'
    # path = env.simulate(start, policy, method)
    # print(path)
    # animate_solution(maze, path)
    