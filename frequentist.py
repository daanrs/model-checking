import stormpy

from main import *

def frequentist(model, measurement):
    return update_mdp(
        model=model, 
        new_matrix = frequentist_create_matrix(model, measurement)
    )


def frequentist_create_matrix(model, measurement):
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
                
                if measurement.has_estimate(state.id, action.id):
                    prob = measurement.get_probability(state.id, action.id, transition.column)
                # if there are no measurements we assume uniform probability
                else:
                    prob = 1 / len(action.transitions)

                builder.add_next_value(act, transition.column, prob)

        nr_actions = nr_actions + len(state.actions)

    matrix = builder.build()

    return matrix
    

if __name__ == "__main__":
    model_file = stormpy.examples.files.prism_mdp_slipgrid
    model_model = stormpy.parse_prism_program(model_file)
    model = stormpy.build_model(model_model)

    measurement = simulate(model)

    model = frequentist(model, measurement)
    print (model)

    print(model.transition_matrix)
