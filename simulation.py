import stormpy
import stormpy.examples
import stormpy.examples.files
import stormpy.simulator
import random

random.seed(23)
path = stormpy.examples.files.prism_mdp_slipgrid
prism_program = stormpy.parse_prism_program(path)

model = stormpy.build_model(prism_program)
simulator = stormpy.simulator.create_simulator(model, seed=42)

def simulate(num_paths = 30, max_path_length = 20):
    paths = []
    for m in range(num_paths):
        path = []
        state, reward, labels = simulator.restart()
        path = [f"{state}"]
        for n in range(max_path_length):
            actions = simulator.available_actions()
            select_action = random.randint(0,len(actions)-1)
            path.append(f"{actions[select_action]}")
            state, reward, labels = simulator.step(actions[select_action])
            path.append(f"{state}")
            if simulator.is_done():
                break
        paths.append(path)
    return paths

print(simulate())
