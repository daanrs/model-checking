import stormpy
import stormpy.examples.files
import stormpy.simulator
import math
from collections import Counter

from simulation import *
from util import *
from frequentist import *
from value_iteration import *


def build_l1_mdp(model, measurement, time, delta):
    number_of_states = len(model.states)
    number_of_actions = max(len(state.actions) for state in model.states)
    c = 14 * number_of_states * math.log(number_of_actions * time / delta)
    distance = {
        (state.id, action.id): math.sqrt(c / max(1, measurement.get_total_frequency(state.id, action.id)))
        for state in model.states
        for action in state.actions
    }
    return distance

def get_supremum(measurement, distance, V, state_id, action_id):
    state_order = sorted(V.keys(), key=lambda x: V[x], reverse=True)
    m = len(state_order)
    probabilities = {}
    probabilities[state_order[0]] = min(1, measurement.get_probability(state_id, action_id, state_order[0], allow_no_estimate=True) + distance[(state_id, action_id)] / 2)
    for j in range(1, m):
        probabilities[state_order[j]] = measurement.get_probability(state_id, action_id, state_order[j], allow_no_estimate=True)
    l = m - 1
    while sum(probabilities.values()) > 1:
        probabilities[state_order[l]] = max(0, 1 - (sum(probabilities.values()) - probabilities[state_order[l]]))
        l -= 1
    
    value = sum(probabilities[state] * V[state] for state in V.keys())
    return value

def value_iteration_next(model, measurement, distance, vs, rewards, gamma):
    vals = { state.id : 
        [
            (
                action.id, 
                (
                    rewards[(state.id, action.id)]
                    + gamma * get_supremum(measurement, distance, vs, state.id, action.id)
                )
            )
            for action in state.actions
        ]
        for state in model.states
    }

    maxes = {
        k: max(v, key=lambda x: x[1])
        for k, v in vals.items()
    }

    return (
        { k: v[0] for k, v in maxes.items() },
        { k: v[1] for k, v in maxes.items() }
    )

def compute_optimistic_policy(model, measurement, distance, gamma, error_bound, rewards):
    max_iter = 200
    vs = { state.id: 0 for state in model.states }
    error = error_bound + 1
    iter = 0
    while (error > error_bound):
        if iter > max_iter:
            raise Exception(f"could not converge within {max_iter} iterations")

        policy, vs_next = value_iteration_next(model, measurement, distance, vs, rewards, gamma)

        error = max(abs(vs_next[i] - vs[i]) for i in vs)
        vs = vs_next

        iter += 1
    return policy

def sample_ucrl2(init_model, measurement, policy):
    sas_counter = Counter() # state-action-state counter
    sa_counter = Counter() # state-action counter
    simulator = stormpy.simulator.create_simulator(init_model, seed=42)
    state, _, _ = simulator.restart()
    time_steps = 0
    trajectories = 1
    while sa_counter[(state, policy[state])] < max(1, measurement.get_total_frequency(state, policy[state])):
        action = policy[state]
        sa_counter[(state, action)] += 1
        next_state, _, _ = simulator.step(action)
        sas_counter[(state, action, next_state)] += 1
        if simulator.is_done():
            state, _, _ = simulator.restart()
            trajectories += 1
        else:
            state = next_state
        time_steps += 1
    return sas_counter, sa_counter, time_steps, trajectories

# gamma: discount factor, error_bound: for value iteration
def ucrl2(init_model, formula, rewards, loops=10, delta=0.1, gamma=0.01, error_bound=0.01):
    time = 1
    data = []
    measurement = Measurement()
    number_of_trajectories = 0

    for k in range(loops):
        # print(f"Total frequencies: {measurement.total_frequencies}")
        distance = build_l1_mdp(init_model, measurement, time, delta)
        optimistic_policy = compute_optimistic_policy(init_model, measurement, distance, gamma, error_bound, rewards)
        sas_counter, sa_counter, time_steps, trajectories = sample_ucrl2(init_model, measurement, optimistic_policy)
        measurement.add_frequencies(sas_counter, sa_counter)
        time += time_steps
        number_of_trajectories += trajectories

        model_dtmc = apply_policy(init_model, optimistic_policy)
        result = stormpy.model_checking(model_dtmc, formula)
        initial_state = model_dtmc.initial_states[0]
        value = result.at(initial_state)
        data.append((number_of_trajectories, value))
    
    ucrl2_mdp = frequentist(init_model, measurement, ucrl2=True)
    l1mdp = (ucrl2_mdp, distance)
    return l1mdp, data


if __name__ == "__main__":
    program = stormpy.parse_prism_program('models/bet_fav.prism')
    # prop = "R=? [F \"goal\"]"
    prop = "R=? [F \"done\"]"
    properties = stormpy.parse_properties(prop, program, None)
    formula = properties[0]
    model = stormpy.build_model(program, properties)

    l1mdp, data = ucrl2(model, formula)
    print(l1mdp[0])
    print(data)
