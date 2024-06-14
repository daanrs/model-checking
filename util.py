import stormpy
import pycarl

from collections import Counter

def update_interval_mdp(model, new_matrix):
    components = stormpy.SparseIntervalModelComponents(
        transition_matrix=new_matrix,
        state_labeling=model.labeling,
    )
    return stormpy.SparseIntervalMdp(components)


def update_interval_from_regular_mdp(model, new_matrix):
    components = stormpy.SparseIntervalModelComponents(
        transition_matrix=new_matrix,
        state_labeling=model.labeling,
    )
    return stormpy.SparseIntervalMdp(components)

    
def update_mdp(model, new_matrix):
    components = stormpy.SparseModelComponents(
        transition_matrix=new_matrix,
        state_labeling=model.labeling,
    )
    return stormpy.SparseMdp(components)


def state_action_rewards_from_model(model):
    rewards = model.reward_models['']
    if rewards.has_state_rewards and rewards.has_state_action_rewards:
        raise Exception('model has both state and state-action rewards')
    elif rewards.has_state_rewards:
        return state_rewards_to_state_action_rewards(model)
    elif rewards.has_state_action_rewards:
        return state_action_rewards(model)
    else:
        raise Exception('model has no rewards')


def state_rewards_to_state_action_rewards(model):
    rewards = model.reward_models['']

    state_actions = [
        (s, a)
        for s in model.states
        for a in s.actions
    ]

    return {
        (state.id, action.id): rewards.get_state_reward(state.id)
        for (state, action) in state_actions
    }


def state_action_rewards(model):
    rewards = model.reward_models['']

    state_actions = [
        (s, a)
        for s in model.states
        for a in s.actions
    ]

    return {
        (state.id, action.id): rewards.get_state_action_reward(i)
        for (i, (state, action)) in enumerate(state_actions)
    }


def storm_state_rewards_from_model_with_policy(model, policy):
    rewards = model.reward_models['']
    if rewards.has_state_rewards:
        return {'': rewards}
    elif rewards.has_state_action_rewards:
        return storm_state_action_rewards_to_state_rewards_with_policy(model, policy)
    else:
        raise Exception('model has no rewards')


def storm_state_action_rewards_to_state_rewards_with_policy(model, policy):
    rewards = model.reward_models['']
    state_actions = [
        (s, a)
        for s in model.states
        for a in s.actions
    ]

    sa_rewards = {
        (state.id, action.id): rewards.get_state_action_reward(i)
        for i, (state, action) in enumerate(state_actions)
    }

    state_rewards = [
        sa_rewards[(state.id, policy[state.id])]
        for state in model.states
    ]

    return {'': stormpy.SparseRewardModel(optional_state_reward_vector=state_rewards)}
    

def create_uMdp_matrix(model, epsilon):
    builder = stormpy.IntervalSparseMatrixBuilder(rows=0, columns=0, entries=0, force_dimensions=False, has_custom_row_grouping=True, row_groups=0)

    nr_actions = 0
    for state in model.states:
        builder.new_row_group(nr_actions)

        for action in state.actions:
            for transition in action.transitions:
                val = transition.value()
                act = action.id + nr_actions
                if val == 0 or val == 1:
                    builder.add_next_value(act, transition.column, pycarl.Interval(val, val))
                else:
                    builder.add_next_value(act, transition.column, pycarl.Interval(epsilon, 1 - epsilon))
        nr_actions = nr_actions + len(state.actions)

    matrix = builder.build()

    return matrix
