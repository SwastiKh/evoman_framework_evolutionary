###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# DEMO : Neuroevolution - Genetic Algorithm  neural network.                  #
# Author: Karine Miras        			                                      #
# karine.smiras@gmail.com     				                                  #
###############################################################################

# imports framework
import sys

from evoman.environment import Environment
from demo_controller import player_controller
from alg_params import *

# imports other libs
import numpy as np
import os
import time
import random


#DIVERSITY OF THE POPULATION AS IMPORTANT AS THE FITNESS OF THE INDIVIDUALS

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)


####   Optimization for controller solution (best genotype-weights for phenotype-network): Ganetic Algorihm    ###

ini = time.time()  # sets time marker


# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

# evaluation
def evaluate(env, x):
    return np.array(list(map(lambda y: simulation(env,y), x)))


def mutation(pop_to_mutate, mut, exit_local_optimum):
    mutated_pop = []
    
    return mutated_pop


def main():
    # choose this for not using visuals and thus making experiments faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"


    # experiment_name = 'optimization_test'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=experiment_name,
                    enemies=[1],
                    playermode="ai",
                    player_controller=player_controller(n_hidden_neurons), # you  can insert your own controller here
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=False)

    env.state_to_log() # checks environment state

    # number of weights for multilayer with 10 hidden neurons
    n_weights = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

    # start writing your own code from here

    if not os.path.exists(experiment_name+'/evoman_solstate'):
        print( '\n NEW EVOLUTION \n')

        # initial population using normal distribution centered around 0
        pop = np.random.normal(mu, sigma, size=(npop, n_weights))
        pop_fit = evaluate(env, pop) #TODO: evaluate function
        best = np.argmax(pop_fit)
        mean = np.mean(pop_fit)
        std = np.std(pop_fit)
        ini_g = 0
        solutions = [pop, pop_fit]
        env.update_solutions(solutions)

    else:
        print( '\n CONTINUING EVOLUTION \n')

        env.load_state()
        pop = env.solutions[0]
        pop_fit = env.solutions[1]

        best = np.argmax(pop_fit)
        mean = np.mean(pop_fit)
        std = np.std(pop_fit)

        # finds last generation number
        file_aux  = open(experiment_name+'/gen.txt','r')
        ini_g = int(file_aux.readline())
        file_aux.close()


if __name__ == '__main__':
    main()