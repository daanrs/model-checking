from operator import itemgetter
import stormpy

from util import *

class X:
    def __init__(self, id, value, lower, upper):
        self.id = id
        self.value = value
        self.lower = lower
        self.upper = upper

    def __lt__(self, other):
        return self.value < other.value

    def get_lower(self):
        return self.lower

    def get_upper(self):
        return self.upper

    def get_prob(self):
        return self.prob

    def is_lower(self):
        self.prob = self.lower

    def is_upper(self):
        self.prob = self.upper

    def set_prob(self, prob):
        self.prob = prob

   
def min_distr(vs, action):
    vals = [
        X(
            id = transition.column,
            value = vs[transition.column],
            lower = transition.value().lower(),
            upper = transition.value().upper()
        )
        for transition in action.transitions
    ]

    return min_distribution(vals)
        

def min_distribution(vs):
    i = 0

    vs = list(sorted(vs))
    limit = sum(v.get_lower() for v in vs)

    while limit - vs[i].get_lower() + vs[i].get_upper() < 1:
        limit = limit - vs[i].get_lower() + vs[i].get_upper()
        vs[i].is_upper()
        i += 1

    vs[i].set_prob(1 - (limit - vs[i].get_lower()))

    for k in range(i + 1, len(vs)):
        vs[k].is_lower()

    return {v.id: v.get_prob() for v in vs}


def next(model, vs, rewards, gamma):
    maxes = [
        [
            (
                action.id, 
                (
                    rewards[(state.id, action.id)]
                    + gamma * sum(
                        transition.value() * vs[transition.column]
                        for transition in action.transitions
                    )
                )
            )
            for action in state.actions
        ]
        for state in model.states
    ]

    return [
        max(m, key = itemgetter(1))
        for m in maxes
    ]

def interval_next(model, vs, rewards, gamma):
    maxes = [
        [
            (
                action.id, 
                (
                    rewards[(state.id, action.id)]
                    + gamma * sum(
                        min_distr(vs, action)[transition.column] * vs[transition.column]
                        for transition in action.transitions
                    )
                )
            )
            for action in state.actions
        ]
        for state in model.states
    ]

    return [
        max(m, key = itemgetter(1))
        for m in maxes
    ]


def next_with_policy(model, policy, vs, rewards, gamma):
    return [
        (
            (action := state.actions[policy[state.id]]).id, 
            (
                rewards[(state.id, action.id)]
                + gamma * sum(
                    transition.value() * vs[transition.column]
                    for transition in action.transitions
                )
            )
        )
        for state in model.states
    ]


def apply_scheduler(model, scheduler):
    builder = stormpy.SparseMatrixBuilder(
        rows=0, 
        columns=0, 
        entries=0, 
        force_dimensions=False, 
        has_custom_row_grouping=False, 
        row_groups=0
    )

    for state in model.states:
        action = state.actions[scheduler[state.id]]
        for transition in action.transitions:
            builder.add_next_value(state.id, transition.column, transition.value())

    matrix = builder.build()

    return stormpy.SparseDtmc(
        stormpy.SparseModelComponents(
            transition_matrix = matrix,
            state_labeling=model.labeling,
            reward_models = model.reward_models
        )
    )
    

def value_iter_with_policy(model, rewards, policy, precision = 0.01, gamma = 0.9, max_iter = 100):
    vs = list(0 for _ in model.states)
    error = 1
    iter = 0

    while (error > precision):
        if iter > max_iter:
            raise Exception(f"could not converge within {max_iter} iterations")

        arg_and_vs = next_with_policy(model, policy, vs, rewards, gamma)

        args = list(a[0] for a in arg_and_vs)
        vs_next = list(a[1] for a in arg_and_vs)

        error = max(abs(vs_next[i] - vs[i]) for i in range(len(vs)))
        vs = vs_next

        iter += 1


    return (args, vs)

def value_iter(model, rewards, precision = 0.01, gamma = 0.9, max_iter = 100):
    vs = list(0 for _ in model.states)
    error = 1
    iter = 0

    while (error > precision):
        if iter > max_iter:
            raise Exception(f"could not converge within {max_iter} iterations")

        arg_and_vs = next(model, vs, rewards, gamma)

        args = list(a[0] for a in arg_and_vs)
        vs_next = list(a[1] for a in arg_and_vs)

        error = max(abs(vs_next[i] - vs[i]) for i in range(len(vs)))
        vs = vs_next

        iter += 1


    return (args, vs)

def interval_value_iter(model, rewards, precision = 0.01, gamma = 0.9, max_iter = 100):
    vs = list(0 for _ in model.states)
    error = 1
    iter = 0

    while (error > precision):
        if iter > max_iter:
            raise Exception(f"could not converge within {max_iter} iterations")

        arg_and_vs = interval_next(model, vs, rewards, gamma)

        args = list(a[0] for a in arg_and_vs)
        vs_next = list(a[1] for a in arg_and_vs)

        error = max(abs(vs_next[i] - vs[i]) for i in range(len(vs)))
        vs = vs_next

        iter += 1


    return (args, vs)
