import stormpy
import pycarl
import math

from main import *

def pac(model, measurement, error_rate = 0.1):
    matrix = pac_create_matrix(model, measurement, error_rate)
    return update_interval_mdp(model, matrix)

def pac_create_matrix(model, measurement, error_rate = 0.1):
    # compute error bounds
    m = 0 # number of successor states with probabilities in (0,1)
    for state in model.states:
        for action in state.actions:
            for transition in action.transitions:
                if 0 < transition.value() < 1:
                    m += 1
    epsilon_M = error_rate / m # stays the same for every (state, action) pair

    builder = stormpy.IntervalSparseMatrixBuilder(rows=0, columns=0, entries=0, force_dimensions=False, has_custom_row_grouping=True, row_groups=0)
    number_actions = 0
    for state in model.states:
        builder.new_row_group(number_actions)
        for action in state.actions:
            N = measurement.get_total_frequency(state.id, action.id)
            delta_M = math.sqrt(math.log(2 / epsilon_M) / (2 * N)) # different for every (state, action) pair

            act = action.id + number_actions
            if (measurement.has_estimate(state.id, action.id)):
                # update probability estimate with PAC interval
                for transition in action.transitions:
                    estimate = measurement.get_probability(state.id, action.id, transition.column)
                    pac_interval = pycarl.Interval(max(0, estimate - delta_M), min(1, estimate + delta_M))
                    builder.add_next_value(act, transition.column, pac_interval)
            else:
                # no probability estimate available, keep previous probability interval
                for transition in action.transitions:
                    builder.add_next_value(act, transition.column, transition.value)
        number_actions += len(state.actions)
    
    transition_matrix = builder.build()

    return transition_matrix

    
if __name__ == "__main__":
    model_file = stormpy.examples.files.prism_mdp_slipgrid
    model_model = stormpy.parse_prism_program(model_file)
    model = stormpy.build_model(model_model)

    measurement = simulate(model)

    pac = pac(model, measurement)
    print(pac)

    print(pac.transition_matrix)
