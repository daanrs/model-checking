import stormpy
import stormpy.examples.files
import pycarl
import random

from collections import Counter
from simulation import simulate
from pac_learning import pac_learning

random.seed(10)

def update_interval_mdp(model, new_matrix):
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


def frequentist_learning(model, frequencies):
    builder = stormpy.SparseMatrixBuilder(rows=0, columns=0, entries=0, force_dimensions=False, has_custom_row_grouping=True, row_groups=0)
    sorted_keys = sorted(frequencies)

    current_state = -1
    current_action = -1
    number_actions = 0
    number_actions_per_state = 0

    for key in sorted_keys:
        state_from, action, state_to = key
        if state_from != current_state:
            number_actions += number_actions_per_state
            number_actions_per_state = 0
            current_action = -1
            print("New group: ", number_actions)
            builder.new_row_group(number_actions)
            current_state = state_from
        if action != current_action:
            number_actions_per_state += 1
            current_action = action
        action += number_actions
        probability = frequencies[key]
        print("Adding value: ", state_from, action, state_to, probability)
        builder.add_next_value(action, state_to, probability)
        
    transition_matrix = builder.build()

    return update_mdp(model=model, new_matrix=transition_matrix)

def create_uMdp_matrix(mdp, epsilon):
    builder = stormpy.IntervalSparseMatrixBuilder(rows=0, columns=0, entries=0, force_dimensions=False, has_custom_row_grouping=True, row_groups=0)

    nr_actions = 0
    for (i, state) in enumerate(mdp.states):
        builder.new_row_group(nr_actions)

        for (act, action) in enumerate(state.actions):
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


maze = stormpy.examples.files.prism_mdp_maze
maze_model = stormpy.parse_prism_program(maze)
maze_final = stormpy.build_model(maze_model)

frequencies = simulate(maze_final)
probabilities = frequencies.probabilities()

mdp = frequentist_learning(maze_final, probabilities)
print(mdp)

uMdp_matrix = create_uMdp_matrix(maze_final, 0.1)
uMdp_model = update_interval_mdp(maze_final, uMdp_matrix)
pac_matrix = pac_learning(uMdp_model, frequencies, 0.1)
pac_imdp = update_interval_mdp(maze_final, pac_matrix)
print(pac_imdp)
