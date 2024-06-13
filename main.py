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
def main(init_model, loops=10):
    data = []
    
    model, strengths = lui_init(init_model)

    # initial data without scheduler
    measurement = simulate(init_model)
    model, strengths = lui_step(model, measurement, strengths)

    rewards = rewards_from_model(init_model)
    # print(rewards)
    
    # with scheduler
    for _ in range(loops):
        policy, _ = interval_value_iter(model, rewards)

        _, real_values = value_iter_with_policy(init_model, rewards, policy)

        initial_state = model.initial_states[0]

        value = real_values[initial_state]
        data.append(value)

        measurement = simulate_policy(init_model, scheduler = policy)
        model, strengths = lui_step(model, measurement, strengths)

    return data


def main2(init_model, loops=10):
    data = []
    
    # initial data without scheduler
    measurement = simulate(init_model)
    model = frequentist(init_model, measurement)
    
    rewards = rewards_from_model(init_model)

    # with scheduler
    for _ in range(loops):
        policy, _ = value_iter(model, rewards)

        _, real_values = value_iter_with_policy(init_model, rewards, policy)

        initial_state = model.initial_states[0]

        value = real_values[initial_state]
        data.append(value)

        measurement = simulate_policy(init_model, , measurement = measurement, scheduler = policy)
        model = frequentist(model, measurement)

    return data


if __name__ == "__main__":
    random.seed(10)

    slipgrid = stormpy.parse_prism_program(stormpy.examples.files.prism_mdp_slipgrid)
    slipgrid_model = stormpy.build_model(slipgrid)

    # data = main2(slipgrid_model)
    data = main(slipgrid_model)
    print(data)

