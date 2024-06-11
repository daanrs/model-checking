import stormpy

from main import *
from simulation import simulate

def lui(model, measurement, strengths):
    pass

model_file = stormpy.examples.files.prism_mdp_coin_2_2
model_model = stormpy.parse_prism_program(model_file)
model = stormpy.build_model(model_model)

frequencies = simulate(model)
probabilities = frequencies.probabilities()

matrix = create_uMdp_matrix(model, 0.2)
new_model = update_interval_mdp(model, matrix)

print(matrix)
print(new_model)
