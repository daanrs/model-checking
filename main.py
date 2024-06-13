import stormpy
import stormpy.examples.files
import pycarl
import random

from util import *
from lui import *
from scheduler import *
from frequentist import *
from simulation import *
from value_iteration import interval_value_iter, value_iter, value_iter_with_policy

# for lui
def main_lui(init_model, loops=10, gamma=0.9, max_iter=1000):
    data = []
    
    model, strengths = lui_init(init_model)

    # initial data without scheduler
    measurement = simulate(init_model)
    model, strengths = lui_step(model, measurement, strengths)

    rewards = state_rewards_from_model(init_model)
    # print(rewards)
    
    # with scheduler
    for _ in range(loops):
        policy, _ = interval_value_iter(model, rewards, gamma=gamma, max_iter=max_iter)

        _, real_values = value_iter_with_policy(init_model, rewards, policy, gamma=gamma, max_iter=max_iter)

        initial_state = model.initial_states[0]

        value = real_values[initial_state]
        data.append(value)

        measurement = simulate_policy(init_model, scheduler = policy)
        model, strengths = lui_step(model, measurement, strengths)

    return data


def main_frequentist(init_model, loops=10, gamma=0.9, max_iter=1000):
    data = []
    
    # initial data without scheduler
    measurement = simulate(init_model)
    model = frequentist(init_model, measurement)
    
    rewards = state_rewards_from_model(init_model)

    # with scheduler
    for _ in range(loops):
        policy, _ = value_iter(model, rewards, gamma=gamma, max_iter=max_iter)

        _, real_values = value_iter_with_policy(init_model, rewards, policy, gamma=gamma, max_iter=max_iter)

        initial_state = model.initial_states[0]

        value = real_values[initial_state]
        data.append(value)

        measurement = simulate_policy(init_model, measurement = measurement, scheduler = policy)
        model = frequentist(model, measurement)

    return data

def main_pac(init_model, loops=10, gamma=0.9, max_iter=1000):
    data = []
    
    model = pac_init(init_model)

    # initial data without scheduler
    measurement = simulate(init_model)
    model = pac_step(model, measurement)

    rewards = state_rewards_from_model(init_model)
    # print(rewards)
    
    # with scheduler
    for _ in range(loops):
        policy, _ = interval_value_iter(model, rewards, gamma=gamma, max_iter=max_iter)

        _, real_values = value_iter_with_policy(init_model, rewards, policy, gamma=gamma, max_iter=max_iter)

        initial_state = model.initial_states[0]

        value = real_values[initial_state]
        data.append(value)

        measurement = simulate_policy(init_model, measurement=measurement, scheduler = policy)
        model = pac_step(model, measurement)

    return data


if __name__ == "__main__":
    random.seed(10)

    # slipgrid = stormpy.parse_prism_program(stormpy.examples.files.prism_mdp_slipgrid)
    # slipgrid_model = stormpy.build_model(slipgrid)

    model = stormpy.build_model(stormpy.parse_prism_program('models/bet_fav.prism'))

    # data = main_frequentist(model, loops=100)
    data = main_lui(model, loops=100)
    # data = main_pac(model)
    print(data)

