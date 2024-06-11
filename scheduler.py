from pac_learning import *

def get_scheduler_for_interval_mdp(model):
    formula = stormpy.parse_properties("Pmax=? [ F \"target\"]")
    initial_state = model.initial_states[0]

    env = stormpy.Environment()
    env.solver_environment.minmax_solver_environment.method = stormpy.MinMaxMethod.value_iteration

    task = stormpy.CheckTask(formula[0].raw_formula, only_initial_states=True)
    task.set_produce_schedulers()
    task.set_robust_uncertainty(True)
    result = stormpy.check_interval_mdp(model, task, env)
    robust_value = result.at(initial_state)
    robust_scheduler = result.scheduler

    task.set_robust_uncertainty(False)
    result = stormpy.check_interval_mdp(model, task, env)
    optimistic_value = result.at(initial_state)
    optimistic_scheduler = result.scheduler
    return robust_scheduler, robust_value, optimistic_scheduler, optimistic_value

if __name__ == "__main__":
    model_file = stormpy.examples.files.prism_mdp_slipgrid
    model_model = stormpy.parse_prism_program(model_file)
    model = stormpy.build_model(model_model)

    measurement = simulate(model)

    pac = pac(model, measurement)
    print(pac.transition_matrix)
    robust_scheduler, robust_value, optimistic_scheduler, optimistic_value = get_scheduler_for_interval_mdp(pac)
    print(robust_value, optimistic_value)
    print(robust_scheduler)
    print(optimistic_scheduler)
