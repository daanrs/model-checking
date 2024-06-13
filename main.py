import stormpy
import stormpy.examples.files
import pycarl
import random

from util import *
from map import *
from lui import *
from scheduler import *
from frequentist import *
from pac_learning import *
from ucrl2 import *
from simulation import *
from value_iteration import *


def main_pac(init_model, paths_per_run, formula, rewards, gamma=0.9, max_iter=1000):
    if len(paths_per_run) < 1:
        raise Exception("empty paths_per_run")

    data = []

    # initial data without scheduler
    model = pac_init(init_model)
    policy = None

    paths_so_far = 0
    measurement = Measurement()

    # with scheduler
    for nr_paths in paths_per_run:
        measurement = simulate_policy(init_model, measurement=measurement, num_paths = nr_paths, policy=policy)
        paths_so_far += nr_paths

        model = pac_step(model, measurement)

        policy, _ = interval_value_iter(model, rewards, gamma=gamma, max_iter=max_iter)
        model_dtmc = apply_policy(init_model, policy)

        result = stormpy.model_checking(model_dtmc, formula)
        initial_state = model.initial_states[0]
        value = result.at(initial_state)

        data.append((paths_so_far, value))

    return data


def main_frequentist(init_model, paths_per_run, formula, rewards, gamma=0.9, max_iter=1000):
    if len(paths_per_run) < 1:
        raise Exception("empty paths_per_run")

    data = []


    policy = None
    paths_so_far = 0
    measurement = Measurement()

    # with scheduler
    for nr_paths in paths_per_run:
        measurement = simulate_policy(init_model, measurement=measurement, num_paths = nr_paths, policy=policy)
        paths_so_far += nr_paths

        model = frequentist(model=init_model, measurement=measurement)

        policy, _ = value_iter(model, rewards, gamma=gamma, max_iter=max_iter)
        model_dtmc = apply_policy(init_model, policy)

        result = stormpy.model_checking(model_dtmc, formula)
        initial_state = model.initial_states[0]
        value = result.at(initial_state)

        data.append((paths_so_far, value))

    return data


def main_map(init_model, paths_per_run, formula, rewards, gamma=0.9, max_iter=1000):
    if len(paths_per_run) < 1:
        raise Exception("empty paths_per_run")

    data = []


    policy = None
    paths_so_far = 0
    measurement = Measurement()
    prior = init_uniform_prior(init_model, 10)

    # with scheduler
    for nr_paths in paths_per_run:
        measurement = simulate_policy(init_model, measurement=measurement, num_paths = nr_paths, policy=policy)
        paths_so_far += nr_paths

        model = map(model=init_model, measurement=measurement, prior=prior)

        policy, _ = value_iter(model, rewards, gamma=gamma, max_iter=max_iter)
        model_dtmc = apply_policy(init_model, policy)

        result = stormpy.model_checking(model_dtmc, formula)
        initial_state = model.initial_states[0]
        value = result.at(initial_state)

        data.append((paths_so_far, value))

    return data


def main_lui(init_model, paths_per_run, formula, rewards, gamma=0.9, max_iter=1000):
    if len(paths_per_run) < 1:
        raise Exception("empty paths_per_run")

    data = []

    # initial data without scheduler
    model, strengths = lui_init(init_model)
    policy = None

    paths_so_far = 0

    # with scheduler
    for nr_paths in paths_per_run:
        measurement = simulate_policy(init_model, num_paths = nr_paths, policy=policy)
        paths_so_far += nr_paths

        model, strengths = lui_step(model, measurement, strengths)

        policy, _ = interval_value_iter(model, rewards, gamma=gamma, max_iter=max_iter)
        model_dtmc = apply_policy(init_model, policy)

        result = stormpy.model_checking(model_dtmc, formula)
        initial_state = model.initial_states[0]
        value = result.at(initial_state)

        data.append((paths_so_far, value))

    return data

def main_ucrl2(init_model, formula, loops=10, gamma=0.9, max_iter=1000):
    l1mdp, data = ucrl2(model, formula, loops=loops, delta=0.1, gamma=gamma, error_bound=0.01)
    return data


if __name__ == "__main__":
    random.seed(10)
    paths_per_run = list(10 * (2**i) for i in range(10))

    # program = stormpy.parse_prism_program('models/bet_fav.prism')
    # prop = "R=? [F \"done\"]"
    # properties = stormpy.parse_properties(prop, program, None)

    # formula=properties[0]

    # model = stormpy.build_model(program, properties)
    # rewards = state_rewards_from_model(model)

    # df1 = {
    #     "map": main_map(model, paths_per_run, formula, rewards=rewards),
    #     "frequentist": main_frequentist(model, paths_per_run, formula, rewards=rewards),
    #     "lui": main_lui(model, paths_per_run=paths_per_run, formula=formula, rewards=rewards),
    #     "pac": main_pac(model, paths_per_run=paths_per_run, formula=formula, rewards=rewards),
    #     # "ucrl2": main_ucrl2(model, paths_per_run, formula)
    # }
    # print(df1)

    # program = stormpy.parse_prism_program('models/bet_unfav.prism')
    # prop = "R=? [F \"done\"]"
    # properties = stormpy.parse_properties(prop, program, None)

    # formula=properties[0]

    # model = stormpy.build_model(program, properties)
    # rewards = state_rewards_from_model(model)

    # df2 = {
    #     "map": main_map(model, paths_per_run, formula, rewards=rewards),
    #     "frequentist": main_frequentist(model, paths_per_run, formula, rewards=rewards),
    #     "lui": main_lui(model, paths_per_run=paths_per_run, formula=formula, rewards=rewards),
    #     "pac": main_pac(model, paths_per_run=paths_per_run, formula=formula, rewards=rewards),
    #     # "ucrl2": main_ucrl2(model, paths_per_run, formula)
    # }
    # print(df2)

    program = stormpy.parse_prism_program('models/bandit.prism')
    prop = "R=? [F \"goal\"]"
    properties = stormpy.parse_properties(prop, program, None)

    formula=properties[0]

    model = stormpy.build_model(program, properties)
    rewards = rewards_from_model(model)

    df3 = {
        "map": main_map(model, paths_per_run, formula, rewards=rewards),
        "frequentist": main_frequentist(model, paths_per_run, formula, rewards=rewards),
        "lui": main_lui(model, paths_per_run=paths_per_run, formula=formula, rewards=rewards),
        "pac": main_pac(model, paths_per_run=paths_per_run, formula=formula, rewards=rewards),
        # "ucrl2": main_ucrl2(model, paths_per_run, formula)
    }

    print(df3)
