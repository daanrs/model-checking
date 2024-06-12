import stormpy
import stormpy.examples.files
import pycarl
import random

from collections import Counter
from simulation import simulate

random.seed(10)

def update_interval_mdp(model, new_matrix):
    components = stormpy.SparseIntervalModelComponents(
        transition_matrix=new_matrix,
        state_labeling=model.labeling,
        reward_models={
            key: create_interval_reward_model(model, model.reward_models[key])
            for key in model.reward_models
        }
    )
    return stormpy.SparseIntervalMdp(components)

    
def update_mdp(model, new_matrix):
    components = stormpy.SparseModelComponents(
        transition_matrix=new_matrix,
        state_labeling=model.labeling,
        reward_models=model.reward_models
    )
    return stormpy.SparseMdp(components)


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
        pycarl.Interval(r, r)
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


if __name__ == "__main__":
    maze = stormpy.examples.files.prism_mdp_maze
    maze_model = stormpy.parse_prism_program(maze)
    maze_final = stormpy.build_model(maze_model)

    slipgrid = stormpy.parse_prism_program(stormpy.examples.files.prism_mdp_slipgrid)
    slipgrid_model = stormpy.build_model(slipgrid)
    print(slipgrid_model.transition_matrix)

    frequencies = simulate(slipgrid_model)
    probabilities = frequencies.probabilities()

    uMdp_matrix = create_uMdp_matrix(slipgrid_model, 0.1)
    uMdp_model = update_interval_mdp(slipgrid_model, uMdp_matrix)
    print(uMdp_model)
    print(uMdp_model.transition_matrix)
