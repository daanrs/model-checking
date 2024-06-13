import stormpy
import stormpy.examples.files
import stormpy.simulator
import math
from collections import Counter

from simulation import *
from util import *
from frequentist import *
from value_iteration import *


def build_l1_mdp(model, measurement, time, delta=0.1):
    number_of_states = len(model.states)
    number_of_actions = max(len(state.actions) for state in model.states)
    c = 14 * number_of_states * math.log(number_of_actions * time / delta)
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
    policy[state_order[0]] = min(1, measurement.get_probability(state_id, action_id, state_order[0], allow_no_estimate=True) + distance[(state_id, action_id)] / 2)
    for j in range(1, m):
        policy[state_order[j]] = measurement.get_probability(state_id, action_id, state_order[j], allow_no_estimate=True)
    l = m - 1
    while sum(policy.values()) > 1:
        policy[state_order[l]] = max(0, 1 - (sum(policy.values()) - policy[state_order[l]]))
        l -= 1
    
    value = sum(policy[state] * V[state] for state in V.keys())
    return policy, value

def compute_optimistic_policy(model, measurement, distance, gamma=0.95, error_bound=0.01):
    model_rewards = state_rewards_from_model(model)
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

def sample_ucrl2(init_model, measurement, policy):
    sas_counter = Counter() # state-action-state counter
    sa_counter = Counter() # state-action counter
    simulator = stormpy.simulator.create_simulator(init_model, seed=42)
    state, _, _ = simulator.restart()
    time_steps = 0
    while sa_counter[(state, policy[state])] < max(1, measurement.get_total_frequency(state, policy[state])):
        action = policy[state]
        sa_counter[(state, action)] += 1
        next_state, _, _ = simulator.step(action)
        sas_counter[(state, action, next_state)] += 1
        state = next_state
        time_steps += 1
    return sas_counter, sa_counter, time_steps

# gamma: discount factor, error_bound: for value iteration
def ucrl2(init_model, formula, number_of_episodes=10, delta=0.1, gamma=0.95, error_bound=0.01):
    time = 1
    data = []

    for k in range(number_of_episodes):
        measurement = Measurement()
        distance = build_l1_mdp(init_model, measurement, time, delta)
        optimistic_policy = compute_optimistic_policy(init_model, measurement, distance, gamma, error_bound)
        sas_counter, sa_counter, time_steps = sample_ucrl2(model, measurement, optimistic_policy)
        measurement.add_frequencies(sas_counter, sa_counter)
        time += time_steps

        policy_as_list = [optimistic_policy[i] for i in range(len(optimistic_policy))]
        model_dtmc = apply_policy(init_model, policy_as_list)
        result = stormpy.model_checking(model_dtmc, formula)
        initial_state = model_dtmc.initial_states[0]
        value = result.at(initial_state)
        data.append(value)
    
    ucrl2_mdp = frequentist(init_model, measurement, ucrl2=True)
    l1mdp = (ucrl2_mdp, distance)
    return l1mdp, data


if __name__ == "__main__":
    program = stormpy.parse_prism_program('models/bet_fav.prism')
    prop = "R=? [F \"done\"]"
    properties = stormpy.parse_properties(prop, program, None)
    formula = properties[0]
    model = stormpy.build_model(program, properties)

    l1mdp, data = ucrl2(model, formula, number_of_episodes=10, delta=0.1, gamma=0.95, error_bound=0.01)
    print(l1mdp[0])
    print(data)
