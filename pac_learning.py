import stormpy
import stormpy.examples
import stormpy.examples.files
import pycarl
import math

# first perform frequentist learning
n_1 = 5
n_2 = 8
n = n_1 + n_2

error_rate = 0.1
m = 1 # number of successor states with probabilities in (0,1)
epsilon_M = error_rate / m
delta_M = math.sqrt(math.log(2 / epsilon_M) / (2 * n))

builder = stormpy.IntervalSparseMatrixBuilder(rows=0, columns=0, entries=0, force_dimensions=False, has_custom_row_grouping=True, row_groups=0)
builder.new_row_group(0)
builder.add_next_value(0, 1, pycarl.core.Interval(n_1 / n - delta_M, n_1 / n + delta_M)) # +- delta_M
builder.add_next_value(0, 2, pycarl.core.Interval(n_2 / n - delta_M, n_2 / n + delta_M)) # +- delta_M
builder.new_row_group(1)
builder.add_next_value(1, 1, pycarl.core.Interval(1, 1))
builder.new_row_group(2)
builder.add_next_value(2, 2, pycarl.core.Interval(1, 1))


transition_matrix = builder.build()

state_labeling = stormpy.storage.StateLabeling(3)
labels = {'init', 'one', 'two'}
for label in labels:
    state_labeling.add_label(label)



components = stormpy.SparseIntervalModelComponents(transition_matrix=transition_matrix, state_labeling=state_labeling)
mdp = stormpy.storage.SparseIntervalMdp(components)
print(mdp)
