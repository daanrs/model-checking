import stormpy
import stormpy.examples.files
import stormpy.simulator
import math
from collections import Counter

from simulation import *
from util import *
from frequentist import *

error_bound = 0.01 # error bound for value iteration
gamma = 0.95 # discount factor
delta = 0.1
time = 1

# for k in range(1, end):

def build_l1_mdp(model, measurement):
    time_k = time
    number_of_states = len(model.states)
    number_of_actions = max(len(state.actions) for state in model.states)
    c = 14 * number_of_states * math.log(number_of_actions * time_k / delta)
    distance = {
        (state.id, action.id): math.sqrt(c / max(1, measurement.get_total_frequency(state.id, action.id)))
        for state in model.states
        for action in state.actions
    }
    return distance

def get_supremum_policy(measurement, distance, V, state_id, action_id):
    state_order = sorted(V.keys(), key=lambda x: V[x], reverse=True)
    m = len(state_order)
    policy = {}
    policy[state_order[0]] = min(1, measurement.get_probability(state_id, action_id, state_order[0]) + distance[(state_id, action_id)] / 2)
    for j in range(1, m):
        policy[state_order[j]] = measurement.get_probability(state_id, action_id, state_order[j])
    l = m
    while sum(policy.values()) > 1:
        policy[state_order[l]] = max(0, 1 - (sum(policy.values()) - policy[state_order[l]]))
        l -= 1
    
    value = sum(policy[state] * V[state] for state in V.keys())
    return policy, value

def compute_optimistic_policy(model, measurement, distance):
    model_rewards = rewards_from_model(model)
    optimistic_policy = {}
    V_old = { state.id: 0 for state in model.states }
    V_new = {}
    while True:
        for state in model.states:
            max_reward = 0
            for action in state.actions:
                policy, value = get_supremum_policy(measurement, distance, V_old, state.id, action.id)
                reward = model_rewards[(state.id, action.id)] + gamma * value
                if reward > max_reward:
                    optimistic_policy[state.id] = action.id
                    max_reward = reward
            V_new[state.id] = max_reward
        if max(abs(V_new[state.id] - V_old[state.id]) for state in model.states) < error_bound:
            break
        V_old = V_new
    return optimistic_policy

def sample_ucrl2(sample_model, measurement, policy):
    sas_counter = Counter() # state-action-state counter
    sa_counter = Counter() # state-action counter
    simulator = stormpy.simulator.create_simulator(sample_model, seed=42)
    state, _, _ = simulator.restart()
    while sa_counter[(state, policy[state])] < max(1, measurement.get_total_frequency(state, policy[state])):
        action = policy[state]
        sa_counter[(state, action)] += 1
        next_state, _, _ = simulator.step(action)
        sas_counter[(state, action, next_state)] += 1
        state = next_state
        # time += 1
    return sas_counter, sa_counter

if __name__ == "__main__":
    model_file = stormpy.examples.files.prism_mdp_slipgrid
    model_model = stormpy.parse_prism_program(model_file)
    model = stormpy.build_model(model_model)

    measurement = Measurement()
    distance = build_l1_mdp(model, measurement)
    optimistic_policy = compute_optimistic_policy(model, measurement, distance)
    sas_counter, sa_counter = sample_ucrl2(model, measurement, optimistic_policy)
    measurement.add_frequencies(sas_counter, sa_counter)

    ucrl2_mdp = frequentist(model, measurement, ucrl2=True)
    print(ucrl2_mdp)
    print(distance)
