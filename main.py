import stormpy
import stormpy.examples.files
import pycarl
import random

from collections import Counter
from simulation import simulate
from frequentist_learning import frequentist_learning

random.seed(10)

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

# print(maze_final)

# m = create_uMdp_matrix(maze_final, 0.2)
# print(m)

# print(maze_final.transition_matrix)

frequencies = simulate(maze_final)
probabilities = frequencies.probabilities()

mdp = frequentist_learning(probabilities)
print(mdp)
