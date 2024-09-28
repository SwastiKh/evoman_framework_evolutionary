import numpy as np
import copy
from alg_params import *
from optimization_dummy import * 


class Population:
    def __init__(self, population_size, individual_size, mutation_rate, fitness_function):
        self.population_size = population_size
        self.individual_size = individual_size
        self.mutation_rate = mutation_rate
        self.fitness_function = fitness_function

        assert population_size > 0
        assert individual_size > 0

        # self.individuals = [Individual(individual_size, fitness_function) for i in range(population_size)]
        self.individuals = np.random.normal(mu, sigma, size=(npop, n_weights))
    def get_best(self):
        self.sort()
        return self.individuals[0]

    def select(self, k):
        weights = np.array([x.score for x in self.individuals])
        weights = weights - min(weights) + 1
        weights = weights / weights.sum()

        parents = np.random.choice(self.individuals, size=k, p=weights)

        return parents

    def crossover(self, parent_1, parent_2):
        child_1 = copy.deepcopy(parent_1)
        child_2 = copy.deepcopy(parent_2)

        pivot = np.random.random_integers(0, self.individual_size)

        child_1.data[:pivot] = parent_2.data[:pivot]
        child_2.data[:pivot] = parent_1.data[:pivot]

        return child_1, child_2

    def mutate(self, child):
        for i in range(child.individual_size):
            if np.random.uniform(0, 1) < self.mutation_rate:
                child.data[i] = (child.data[i] + 1) % 2

    def sort(self):
        self.individuals = sorted(self.individuals, key=lambda x: x.score, reverse=True)
        self.individuals = self.individuals[:self.population_size]

    def run(self):
        self.sort()

        parent_1, parent_2 = self.select(2)

        child_1, child_2 = self.crossover(parent_1, parent_2)

        self.mutate(child_1)
        self.mutate(child_2)

        child_1.reevaluate()
        child_2.reevaluate()

        self.individuals.append(child_1)
        self.individuals.append(child_2)
