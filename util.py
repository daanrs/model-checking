import stormpy
import pycarl

from collections import Counter
from simulation import simulate

def update_interval_mdp(model, new_matrix):
    components = stormpy.SparseIntervalModelComponents(
        transition_matrix=new_matrix,
        state_labeling=model.labeling,
        # reward_models=model.reward_models
    )
    return stormpy.SparseIntervalMdp(components)


def update_interval_from_regular_mdp(model, new_matrix):
    components = stormpy.SparseIntervalModelComponents(
        transition_matrix=new_matrix,
        state_labeling=model.labeling,
        # reward_models={
        #     key: create_interval_reward_model(model, model.reward_models[key])
        #     for key in model.reward_models
        # }
    )
    return stormpy.SparseIntervalMdp(components)

    
def update_mdp(model, new_matrix):
    components = stormpy.SparseModelComponents(
        transition_matrix=new_matrix,
        state_labeling=model.labeling,
        # reward_models=model.reward_models
    )
    return stormpy.SparseMdp(components)


def state_rewards_from_model(model):
    rewards = model.reward_models['']
    if not rewards.has_state_rewards:
        raise Exception("we only deal with state-action rewards")

    state_actions = [
        (s, a)
        for s in model.states
        for a in s.actions
    ]

    return {
        (state.id, action.id): rewards.get_state_reward(i)
        for i, (state, action) in enumerate(state_actions)
    }

def rewards_from_model(model):
    rewards = model.reward_models['']
    if rewards.has_state_rewards or not rewards.has_state_action_rewards:
        raise Exception("we only deal with state-action rewards")

    state_actions = [
        (s, a)
        for s in model.states
        for a in s.actions
    ]

    return {
        (state.id, action.id): rewards.get_state_action_reward(i)
        for i, (state, action) in enumerate(state_actions)
    }
    

def create_interval_reward_model(model, rewards):
    if rewards.has_state_rewards or not rewards.has_state_action_rewards:
        raise Exception("we only deal with state-action rewards")

    state_actions = [
        (s, a)
        for s in model.states
        for a in s.actions
    ]

    state_action_rewards = [
        rewards.get_state_action_reward(i)
        for i, _ in enumerate(state_actions)
    ]

    state_action_intervals = [
        pycarl.Interval(r)
        for r in state_action_rewards
    ]

    return stormpy.SparseIntervalRewardModel(optional_state_action_reward_vector=state_action_intervals)


def create_uMdp_matrix(model, epsilon):
    builder = stormpy.IntervalSparseMatrixBuilder(rows=0, columns=0, entries=0, force_dimensions=False, has_custom_row_grouping=True, row_groups=0)

    nr_actions = 0
    for state in model.states:
        builder.new_row_group(nr_actions)

        for action in state.actions:
            for transition in action.transitions:
                val = transition.value()
                act = action.id + nr_actions
                if val == 1:
                    builder.add_next_value(act, transition.column, pycarl.Interval(1, 1))
                else:
                    builder.add_next_value(act, transition.column, pycarl.Interval(epsilon, 1 - epsilon))
        nr_actions = nr_actions + len(state.actions)

    matrix = builder.build()

    return matrix
