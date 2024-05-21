import stormpy
import stormpy.examples
import stormpy.examples.files

path = stormpy.examples.files.prism_dtmc_die
prism_program = stormpy.parse_prism_program(path)
model = stormpy.build_model(prism_program)

n_1 = 5
n_2 = 8
n = n_1 + n_2

builder = stormpy.SparseMatrixBuilder(rows=0, columns=0, entries=0, force_dimensions=False, has_custom_row_grouping=True, row_groups=0)
builder.new_row_group(0)
builder.add_next_value(0, 1, n_1 / n)
builder.add_next_value(0, 2, n_2 / n)
builder.new_row_group(1)
builder.add_next_value(1, 1, 1)
builder.new_row_group(2)
builder.add_next_value(2, 2, 1)


transition_matrix = builder.build()

state_labeling = stormpy.storage.StateLabeling(3)
labels = {'init', 'one', 'two'}
for label in labels:
    state_labeling.add_label(label)



components = stormpy.SparseModelComponents(transition_matrix=transition_matrix, state_labeling=state_labeling)
mdp = stormpy.storage.SparseMdp(components)
print(mdp)
