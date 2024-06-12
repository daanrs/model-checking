import stormpy
import stormpy.examples.files
import math
from simulation import *

delta = 0.1
t = 1
measurement = Measurement()

# for k in range(1, end):

def build_l1_mdp(model):
    t_k = t
    number_of_states = len(model.states)
    number_of_actions = max(len(state.actions) for state in model.states)
    c = 14 * number_of_states * math.log(number_of_actions * t_k / delta)
    d = {
        (state.id, action.id): math.sqrt(c / max(1, measurement.get_total_frequency(state.id, action.id)))
        for state in model.states
        for action in state.actions
    }
    print(d)

if __name__ == "__main__":
    model_file = stormpy.examples.files.prism_mdp_slipgrid
    model_model = stormpy.parse_prism_program(model_file)
    model = stormpy.build_model(model_model)

    # measurement = simulate(model)
    build_l1_mdp(model)
