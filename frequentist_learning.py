import stormpy

def build_matrix_from_data(frequencies):
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
    number_states = transition_matrix.nr_columns

    state_labeling = stormpy.storage.StateLabeling(number_states)
    labels = ( f"state_{i}" for i in range(0, number_states))
    for label in labels:
        state_labeling.add_label(label)

    components = stormpy.SparseModelComponents(transition_matrix=transition_matrix, state_labeling=state_labeling)
    mdp = stormpy.storage.SparseMdp(components)
    return mdp
