import stormpy
import stormpy.examples
import stormpy.examples.files
import stormpy.simulator
import random
from collections import Counter

random.seed(23)
path = stormpy.examples.files.prism_mdp_slipgrid
prism_program = stormpy.parse_prism_program(path)

model = stormpy.build_model(prism_program)
simulator = stormpy.simulator.create_simulator(model, seed=42)

def simulate(num_paths = 30, max_path_length = 20):
    frequencies = Counter() # (state_from, action, state_to) -> frequency
    total_frequencies = Counter() # (state, action) -> frequency
    for m in range(num_paths):
        state_from, reward, labels = simulator.restart()
        for n in range(max_path_length):
            actions = simulator.available_actions()
            select_action = random.randint(0,len(actions)-1)
            state_to, reward, labels = simulator.step(actions[select_action])
            frequencies[(state_from, actions[select_action], state_to)] += 1
            total_frequencies[(state_from, actions[select_action])] += 1
            if simulator.is_done():
                break
            state_from = state_to
    
    for key in frequencies:
        frequencies[key] /= total_frequencies[(key[0], key[1])] # turn freuqencies into probabilities
    return frequencies

frequencies = simulate()
print(frequencies)
