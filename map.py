import stormpy
import stormpy.examples.files

from util import *

def init_uniform_prior(model, alpha):
    return {
        (state.id, action.id, transition.column): alpha
        for state in model.states
        for action in state.actions
        for transition in action.transitions
    }


def map(model, measurement, prior):
    return update_mdp(model, map_create_matrix(model, measurement, prior))


def map_create_matrix(model, measurement, prior):
    builder = stormpy.SparseMatrixBuilder(
        rows=0, 
        columns=0, 
        entries=0, 
        force_dimensions=False, 
        has_custom_row_grouping=True, 
        row_groups=0
    )

    posterior = {
        (state.id, action.id, transition.column) :
            prior[(state.id, action.id, transition.column)]
            + measurement.get_frequency(state.id, action.id, transition.column)
        for state in model.states
        for action in state.actions
        for transition in action.transitions
    }
    
    nr_actions = 0
    for state in model.states:
        builder.new_row_group(nr_actions)

        for action in state.actions:
            for transition in action.transitions:
                act = action.id + nr_actions
                
                prob = (
                    (
                        posterior[(state.id, action.id, transition.column)] - 1
                    ) 
                    / (
                        sum(
                            posterior[(state.id, action.id, alpha.column)]
                            for alpha in action.transitions
                        ) 
                        - len(action.transitions)
                    )
                )
                builder.add_next_value(act, transition.column, prob)

        nr_actions = nr_actions + len(state.actions)

    matrix = builder.build()

    return matrix
