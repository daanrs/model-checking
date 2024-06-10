import stormpy
import stormpy.simulator
import random
from collections import Counter

def simulate(model, num_paths = 100, max_path_length = 200):
    stop_next_round = False
    simulator = stormpy.simulator.create_simulator(model, seed=42)
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
            state_from = state_to
            if stop_next_round:
                stop_next_round = False
                break
            if simulator.is_done():
                stop_next_round = True
    
    for key in frequencies:
        frequencies[key] /= total_frequencies[(key[0], key[1])] # turn frequencies into probabilities
    return frequencies
