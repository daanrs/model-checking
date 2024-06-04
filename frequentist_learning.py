import stormpy
import stormpy.examples
import stormpy.examples.files

def build_matrix_from_data(x):
    builder = stormpy.SparseMatrixBuilder(rows=0, columns=0, entries=0, force_dimensions=False, has_custom_row_grouping=True, row_groups=0)


    current_state = x[0, 0]
    builder.new_row_group(current_state)
    nr_states = 1

    for row in x:
        if row[0] != current_state:
            builder.new_row_group(row[0])
            current_state = row[0]
            nr_states += 1
        builder.add_nerowt_value(row[1], row[2], row[3])
        
    transition_matrix = builder.build()

    state_labeling = stormpy.storage.StateLabeling(nr_states)
    labels = ( f"state_{i}" for i in range(0, nr_states))
    for label in labels:
        state_labeling.add_label(label)



components = stormpy.SparseModelComponents(transition_matrix=transition_matrix, state_labeling=state_labeling)
mdp = stormpy.storage.SparseMdp(components)
print(mdp)
