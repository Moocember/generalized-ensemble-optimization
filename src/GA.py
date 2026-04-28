import pandas as pd
import numpy as np
from hyperopt import hp,pyll,fmin, tpe,STATUS_OK,space_eval
from Load_Data import load_numerai_data, load_optimized_models,pickleIt
from functools import partial
from ML import Ensemble,SGDClassify,XGB,SVC,LogisticRegression
from Bayes import fitness, classification, binaryAccuracy,classify_models,subset_keys
from pyeasyga import pyeasyga
import random
from operator import attrgetter
import copy

def spawn(params):
    """
    Creates a random model with random hyperparameters

    Args:
        params: Parameter space to select from.

    Returns:
        Random model and hyperparameter set.
    """
    return pyll.stochastic.sample(params)['models']

def randomLengthSample(sample):
    """
    Subsample of random length

    Args:
        sample: Data to subsample from.

    Returns:
        Subsample of random length.
    """
    num_samples = random.randint(1,len(sample)-1)
    return random.sample(sample,num_samples)



def update_probabilities(data,blender_probabilities,params):
    """
    Appends an additional model's predictions to blender_probabilities.

    Args:
        data: Train, CV and test data.
        blender_probabilities: Probabilities of all models in the existing ensemble.
        params: Parameters of the new model.

    Returns:
        Indivual model probabilities including the new model.
    """
    probabilities = classification(params, data)

    new_probabilities = dict()
    for key in probabilities.keys():
        column_name = blender_probabilities[key].columns.values
        new_column_name = np.r_[column_name,np.asarray((len(column_name),))]
        new_probabilities[key] = pd.concat([blender_probabilities[key], pd.DataFrame(probabilities[key])], axis=1)
        new_probabilities[key].columns = new_column_name
    return new_probabilities



def update_blender_probabilities(data,blender_probabilities,params):
    """
    Updates ensemble prediction probabilities to include an additional model.

    Args:
        data: Train, CV and test data.
        blender_probabilities: Probabilities of all models in the existing ensemble.
        params: Parameters of the new model.

    Returns:
        Ensemble probabilities including the new model.
    """

    data_copy = data.copy()

    new_probabilities = update_probabilities(data_copy, blender_probabilities, params)

    data_copy.update(new_probabilities)

    new_blended_probabilities = classification(blender_probabilities['params'], data_copy)

    for target in new_blended_probabilities.keys():
        idx = target[1:]
        new_probabilities['AUC' + idx] = binaryAccuracy(new_blended_probabilities['x' + idx], data_copy['y' + idx])

    new_probabilities['params'] = blender_probabilities['params']

    return new_probabilities

class GeneticEnsemble(pyeasyga.GeneticAlgorithm):
    def __init__(self,
                 data,
                 blender_probabilities,
                 params,
                 population_size=50,
                 generations=100,
                 crossover_probability=0.8,
                 mutation_probability=0.2,
                 elitism=True,
                 maximise_fitness=True):
        """
        Child of GeneticAlgorihtm class from pyeasyga.
        This child class over rides method and class functions from the GeneticAlgorithm class.
        Sections of this code are duplicated from the code within the pyeasyga library and modified.

        For more info about the pyeasyga package: https://pyeasyga.readthedocs.io/en/latest/usage.html

        Args:
                data: Train, CV and test data.
                blender_probabilities: Probabilities of all models in the existing ensemble.
                params: Parameter space of models.
                population_size: Number of agents in population
                generations: Number of generations: how many times to run the algorithm
                crossover_probability: Percentage chance of two surviving algorithms breeding
                mutation_probability: Percentage chance of an algorithm mutating.
                elitism: If true, the top solution will always survive
                maximise_fitness: If true, algorithm will aim to maximize the fitness function.
        """

        self.data = data
        self.blender_probabilities = blender_probabilities
        self.seed_data = params
        self.population_size = population_size
        self.generations = generations
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.elitism = elitism
        self.maximise_fitness = maximise_fitness

        self.current_generation = []

        def fitness(data, blender_probabilities, params):
            """
            The genetic algorithm aims to maximize this fitness function.
            Trains a model, adds its predictions to the ensemble and returns the marginal AUC.

            Args:
                data: Train, CV and test data.
                blender_probabilities: Probabilities of all models in the existing ensemble.
                params: Parameters of the new model.

            Returns:
                Marginal binary accuracy of ensemble with new model.
            """
            new_blender_probabilities = update_blender_probabilities(data,blender_probabilities,params)
            marginal_AUC = new_blender_probabilities['AUC_validation'] - blender_probabilities['AUC_validation']
            print('Marginal AUC: ', marginal_AUC, new_blender_probabilities['AUC_validation'])
            return marginal_AUC

        #def create_individual(params):
        #    return spawn(params)

        def crossover(parent_1, parent_2):
            """
            Crossbreeds two surviving agents: children agents take parameters from the two parents.

            Args:
                parent_1: Model parameters of a surviving agent.
                parent_2: Model parameters of another surviving agent.

            Returns:
                Two children sets of parameters to be added to the next generation's population.
            """
            keywords = parent_1['optimized']
            if keywords == parent_2['optimized'] and len(keywords) > 1:
                index = randomLengthSample(keywords)
                child_1 = parent_1.copy()
                child_2 = parent_2.copy()
                for key in keywords:
                    if key in index:
                        child_1[key] = parent_2[key]
                        child_1['param'][key] = parent_2[key]
                    else:
                        child_2[key] = parent_1[key]
                        child_2['param'][key] = parent_1[key]
                return child_1, child_2
            return parent_1, parent_2

        def mutate(self,child):
            """
            Mutation of child agents: Parameters are randomly changed to explore the search space.

            Args:
                self: GeneticEnsemble class self.
                child: Model parameters of a newly created child agent.

            Returns:
                Two children sets of parameters to be added to the next generation's population.
            """
            keywords = child['optimized']
            index = randomLengthSample(keywords)
            mutated = spawn(self.seed_data)
            if mutated['algo'] == child['algo']:
                for i in index:
                    child[i] = mutated[i]
                    child['param'][i] = mutated[i]
            return child

        def tournament_selection(population):
            """
            This function was created in pyeasyga's GeneticAlgorithm Class.
            It must be included because it lies within the __init__ function which is being overridden by this child class.

            For more info about the pyeasyga package: https://pyeasyga.readthedocs.io/en/latest/usage.html

            Select a random number of individuals from the population and return the fittest member of them all.

            Args:
                population: All living agents' parameters.

            Returns:
                Most fit of randomly selected subset of population.
            """
            if self.tournament_size == 0:
                self.tournament_size = 2
            members = random.sample(population, self.tournament_size)
            members.sort(
                key=attrgetter('fitness'), reverse=self.maximise_fitness)
            return members[0]

        self.fitness_function = fitness
        self.tournament_selection = tournament_selection
        self.tournament_size = self.population_size // 10
        self.random_selection = random.choice
        self.create_individual = spawn
        self.crossover_function = crossover
        self.mutate_function = mutate
        self.selection_function = self.tournament_selection

    def calculate_population_fitness(self):
        """
        This function was modified from in pyeasyga's GeneticAlgorithm Class.
        It must be included because it lies within the __init__ function which is being overridden by this child class.

        For more info about the pyeasyga package: https://pyeasyga.readthedocs.io/en/latest/usage.html

        Changed this function from the pyeasyga library to allow the data and blender_probabilities to pass through to the fitness function

        Args:
                self: GeneticEnsemble class self.

        Returns:
            None
        """
        for individual in self.current_generation:
            individual.fitness = self.fitness_function(self.data,self.blender_probabilities,individual.genes)

    def rank_population(self):
        """
        This function was modified from pyeasyga's GeneticAlgorithm Class.
        It must be included because it lies within the __init__ function which is being overridden by this child class.

        For more info about the pyeasyga package: https://pyeasyga.readthedocs.io/en/latest/usage.html

        Changed this function from the pyeasyga library to add the most fit model to the ensemble if it has a positive marginal AUC.
        Sort the population by fitness according to the order defined by maximise_fitness.

        Args:
            self: GeneticEnsemble class self.

        Returns:
            None

        """
        print(self.current_generation[0].genes)
        self.current_generation.sort(
            key=attrgetter('fitness'), reverse=self.maximise_fitness)
        most_fit = update_blender_probabilities(self.data, self.blender_probabilities, self.current_generation[0].genes)

        if most_fit['AUC_validation'] - self.blender_probabilities['AUC_validation'] > 0.00001:
            self.blender_probabilities.update(most_fit)
            pickleIt(most_fit, 'blend_params3')
        else:
            quit()

        print('Ensemble Fitness:', most_fit['AUC_validation'],most_fit['AUC_test'], len(most_fit['x_train'].T))

    def create_new_population(self):
        """
        This function was created in pyeasyga's GeneticAlgorithm Class.
        It must be included because it lies within the __init__ function which is being overridden by this child class.

        For more info about the pyeasyga package: https://pyeasyga.readthedocs.io/en/latest/usage.html

        Create a new population using the genetic operators (selection, crossover, and mutation) supplied.

        Args:
            self: GeneticEnsemble class self.

        Returns:
            None
        """
        new_population = []
        elite = copy.deepcopy(self.current_generation[0])
        selection = self.selection_function

        while len(new_population) < self.population_size:
            parent_1 = copy.deepcopy(selection(self.current_generation))
            parent_2 = copy.deepcopy(selection(self.current_generation))

            child_1, child_2 = parent_1, parent_2
            child_1.fitness, child_2.fitness = 0, 0

            can_crossover = random.random() < self.crossover_probability
            can_mutate = random.random() < self.mutation_probability

            if can_crossover:
                child_1.genes, child_2.genes = self.crossover_function(
                    parent_1.genes, parent_2.genes)

            if can_mutate:
                self.mutate_function(self,child_1.genes)
                self.mutate_function(self,child_2.genes)

            new_population.append(child_1)
            if len(new_population) < self.population_size:
                new_population.append(child_2)

        if self.elitism:
            new_population[0] = elite

        self.current_generation = new_population



if __name__ == '__main__':

    data = load_numerai_data()

    blender_probabilities = load_optimized_models('blend_params2.pickle')

    sgd = SGDClassify(alpha=hp.uniform('alpha', 0, 1),
                      max_iter=hp.uniform('max_iter', 1000, 5000))

    xgb = XGB(n_estimators= hp.choice('n_estimators', np.arange(100, 1000, dtype=int)),
              eta = hp.uniform('eta', 0.025, 0.5),
              max_depth = hp.choice('max_depth', np.arange(1, 14, dtype=int)),
              min_child_weight = hp.quniform('min_child_weight', 1, 6, 1),
              subsample = hp.uniform('subsample', 0.5, 1),
              gamma = hp.uniform('gamma', 0.5, 1),
              colsample_bytree = hp.uniform('colsample_bytree', 0.5, 1))

    svc = SVC(C = hp.lognormal('C',0,20),
              max_iter=hp.uniform('max_iter', 1000, 5000))

    models = {'SGD': sgd,
              'SVC': svc,
              'XGB': xgb}

    ensemble = Ensemble(models).__dict__

    GA = GeneticEnsemble(data = data,
                         blender_probabilities = blender_probabilities,
                         params = ensemble,
                         population_size=30,
                         generations=20,
                         crossover_probability=.6,
                         mutation_probability=.6,
                         elitism=True,
                         maximise_fitness=True)

    GA.run()

    for individual in GA.last_generation():
        print(individual)