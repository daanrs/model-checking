import stormpy
import stormpy.examples.files

from util import *
from simulation import simulate

def init_strengths(model, lower, upper):
    return {
        (state.id, action.id, transition.column): (lower, upper)
        for state in model.states
        for action in state.actions
        for transition in action.transitions
    }

def lui_init(model, epsilon = 0.2 , lower=0, upper=10):
    matrix = create_uMdp_matrix(model, epsilon)
    strengths = init_strengths(model, lower, upper)
    return update_interval_from_regular_mdp(model, matrix), strengths


def lui_step(model, measurement, strengths):
    matrix, strengths = lui_create_matrix(model, measurement, strengths)
    return update_interval_mdp(model, matrix), strengths


def lui_create_matrix(model, measurement, strengths):
    builder = stormpy.IntervalSparseMatrixBuilder(
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
            if measurement.has_estimate(state.id, action.id):
                lower_agreement = all(
                    measurement.get_probability(state.id, action.id, next.column) >= next.value().lower()
                    for next in action.transitions
                )
                upper_agreement = all(
                    measurement.get_probability(state.id, action.id, next.column)  <= next.value().upper()
                    for next in action.transitions
                )

                for transition in action.transitions:
                    strength_lower, strength_upper = strengths[(state.id, action.id, transition.column)]

                    k = measurement.get_frequency(state.id, action.id, transition.column)
                    k_total = measurement.get_total_frequency(state.id, action.id)

                    lower = transition.value().lower()
                    if lower_agreement:
                        next_lower = (  strength_upper  * lower  + k ) / ( strength_upper + k_total ) 
                    else:
                        next_lower = (  strength_lower * lower  + k ) / ( strength_lower + k_total ) 

                    upper = transition.value().upper()

                    if upper_agreement:
                        next_upper = (  strength_upper  * upper + k ) / ( strength_upper + k_total ) 
                    else:
                        next_upper = (  strength_lower * upper + k ) / ( strength_lower + k_total ) 

                    strengths[(state.id, action.id, transition.column)] = (
                        strength_lower + k_total,
                        strength_upper + k_total
                    )

                    interval = pycarl.Interval(next_lower, next_upper)

                    act = action.id + nr_actions
                    builder.add_next_value(act, transition.column, interval)
            else:
                for transition in action.transitions:
                    act = action.id + nr_actions
                    builder.add_next_value(act, transition.column, transition.value)

        nr_actions = nr_actions + len(state.actions)

    matrix = builder.build()

    return matrix, strengths
    
if __name__ == "__main__":
    model_file = stormpy.examples.files.prism_mdp_slipgrid
    model_model = stormpy.parse_prism_program(model_file)
    model = stormpy.build_model(model_model)

    measurement = simulate(model)

    lui, strengths = lui_init(model, 0.2, 0, 100)
    print(lui.transition_matrix)

    lui_next, strengths = lui_step(lui, measurement, strengths)
    print(lui_next.transition_matrix)
