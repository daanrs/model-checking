import stormpy
import stormpy.examples.files
import pycarl
import random

from util import *


if __name__ == "__main__":
    random.seed(10)
    maze = stormpy.examples.files.prism_mdp_maze
    maze_model = stormpy.parse_prism_program(maze)
    maze_final = stormpy.build_model(maze_model)

    slipgrid = stormpy.parse_prism_program(stormpy.examples.files.prism_mdp_slipgrid)
    slipgrid_model = stormpy.build_model(slipgrid)
    print(slipgrid_model.transition_matrix)

