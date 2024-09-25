
# dummy parameters
dom_u = 1
dom_l = -1
mu = 0
sigma = 0.21
npop = 100
gens = 30
mutation = 0.2
last_best = 0

n_hidden_neurons = 128

#  Other parameters
seed = 1234
experiment_name = 'crossover_baseline_enemy1_'+str(seed)
run_mode = 'train' # train or test


alg_args = {'experiment_name': experiment_name, 'seed': seed, 'dom_u': dom_u, 'dom_l': dom_l, 'npop': npop, 'gens': gens, 'mutation': mutation, 'last_best': last_best}