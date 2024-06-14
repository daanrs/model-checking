import sys
import argparse
import random
from main import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reinforcement Learning Algorithms')
    parser.add_argument('model_path', metavar='prism_model', type=str, nargs=1,
                        help='path to a file containing a PRISM model')
    parser.add_argument('target_label', metavar='target_label', type=str, nargs=1,
                        help='target label in the PRISM model')

    args = parser.parse_args()

    random.seed(10)
    paths_per_run = list(10 * (2**i) for i in range(10))
    gamma = 0.01

    program = stormpy.parse_prism_program(args.model_path[0])
    prop = f"R=? [F \"{args.target_label[0]}\"]"
    properties = stormpy.parse_properties(prop, program, None)

    formula=properties[0]

    model = stormpy.build_model(program, properties)
    rewards = state_action_rewards_from_model(model)

    df = {
        "map": main_map(model, paths_per_run, formula=formula, gamma = gamma, rewards=rewards),
        "frequentist": main_frequentist(model, paths_per_run, formula=formula, gamma = gamma, rewards=rewards),
        "lui": main_lui(model, paths_per_run=paths_per_run, formula=formula, gamma = gamma, rewards=rewards),
        "pac": main_pac(model, paths_per_run=paths_per_run, formula=formula, gamma = gamma, rewards=rewards),
        "ucrl2": main_ucrl2(model, loops=15, formula=formula, gamma=gamma, rewards=rewards)
    }

    json.dump(df, sys.stdout)
