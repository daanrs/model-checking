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


def frequentist(model, measurement):
    return update_mdp(
        model=model, 
        new_matrix = create_matrix(model, measurement)
    )


def create_matrix(model, measurement):
    
    builder = stormpy.SparseMatrixBuilder(
        rows=0, 
        columns=0, 
        entries=0, 
        force_dimensions=False, 
        has_custom_row_grouping=True, 
        row_groups=0
    )

    nr_actions = 0
    for state in model.states:
        builder.new_row_group(nr_actions)

        for action in state.actions:
            for transition in action.transitions:
                act = action.id + nr_actions
                
                prob = measurement.get_probability(state.id, action.id, transition.column)
                builder.add_next_value(act, transition.column, prob)

        nr_actions = nr_actions + len(state.actions)

    matrix = builder.build()

    return matrix
    

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


maze = stormpy.examples.files.prism_mdp_maze
maze_model = stormpy.parse_prism_program(maze)
maze_final = stormpy.build_model(maze_model)

frequencies = simulate(maze_final)
probabilities = frequencies.probabilities()

uMdp_matrix = create_uMdp_matrix(maze_final, 0.1)
uMdp_model = update_interval_mdp(maze_final, uMdp_matrix)
pac_matrix = pac_learning(uMdp_model, frequencies, 0.1)
pac_imdp = update_interval_mdp(maze_final, pac_matrix)
print(pac_imdp)
