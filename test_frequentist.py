import stormpy
import stormpy.examples
import stormpy.examples.files
import random

from collections import Counter
from simulation import simulate
from frequentist_learning import build_matrix_from_data

random.seed(23)
path = stormpy.examples.files.prism_mdp_slipgrid
prism_program = stormpy.parse_prism_program(path)

model = stormpy.build_model(prism_program)

frequencies = simulate(model)
print(frequencies)

mdp = build_matrix_from_data(frequencies)
print(mdp)
