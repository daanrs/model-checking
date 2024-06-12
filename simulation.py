import stormpy
import stormpy.simulator
import random
from collections import Counter

class Measurement:
    def __init__(self, frequencies = Counter(), total_frequencies = Counter()):
        self.frequencies = frequencies
        self.total_frequencies = total_frequencies

    def measure(self, state_from, action, state_to):
        self.frequencies[(state_from, action, state_to)] += 1
        self.total_frequencies[(state_from, action)] += 1

    def has_estimate(self, state_from, action):
        return self.total_frequencies[(state_from, action)] != 0

    def get_frequency(self, state_from, action, state_to):
        return self.frequencies[(state_from, action, state_to)]

    def get_total_frequency(self, state_from, action):
        return self.total_frequencies[(state_from, action)]

    def get_probability(self, state_from, action, state_to):
        if not self.has_estimate(state_from, action):
            return 0
            # raise Exception("Tried to get probability for a state-action pair which has never been measured")
        else:
            return (
                self.frequencies[(state_from, action, state_to)] 
                / self.total_frequencies[(state_from, action)]
            )
    
def simulate_policy(model, num_paths = 100, max_path_length = 200, measurement = Measurement(), scheduler = None):
    stop_next_round = False
    simulator = stormpy.simulator.create_simulator(model, seed=42)
    for m in range(num_paths):
        state_from, reward, labels = simulator.restart()
        for n in range(max_path_length):
            if scheduler:
                action = scheduler[state_from]
            else:
                actions = simulator.available_actions()
                select_action = random.randint(0,len(actions)-1)
                action = actions[select_action]

            state_to, reward, labels = simulator.step(action)
            measurement.measure(state_from, action, state_to)

            state_from = state_to
            if stop_next_round:
                stop_next_round = False
                break
            if simulator.is_done():
                stop_next_round = True

    return measurement


def simulate(model, num_paths = 100, max_path_length = 200, measurement = Measurement(), scheduler = None):
    stop_next_round = False
    simulator = stormpy.simulator.create_simulator(model, seed=42)
    for m in range(num_paths):
        state_from, reward, labels = simulator.restart()
        for n in range(max_path_length):
            if scheduler:
                choice = scheduler.get_choice(state_from)
                action = choice.get_deterministic_choice()
            else:
                actions = simulator.available_actions()
                select_action = random.randint(0,len(actions)-1)
                action = actions[select_action]

            state_to, reward, labels = simulator.step(action)
            measurement.measure(state_from, action, state_to)

            state_from = state_to
            if stop_next_round:
                stop_next_round = False
                break
            if simulator.is_done():
                stop_next_round = True

    return measurement
