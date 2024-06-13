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

    while i < len(vs) and (limit - vs[i].get_lower() + vs[i].get_upper() < 1):
        limit = limit - vs[i].get_lower() + vs[i].get_upper()
        vs[i].is_upper()
        i += 1

    if i < len(vs):
        vs[i].set_prob(1 - (limit - vs[i].get_lower()))

        for k in range(i + 1, len(vs)):
            vs[k].is_lower()

    return {v.id: v.get_prob() for v in vs}


def next(model, vs, rewards, gamma):
    vals = { state.id : 
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
    }

    maxes = {
        k: max(v, key=lambda x: x[1])
        for k, v in vals.items()
    }

    return (
        { k: v[0] for k, v in maxes.items() },
        { k: v[1] for k, v in maxes.items() }
    )

def interval_next(model, vs, rewards, gamma):
    vals = { state.id : 
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
    }

    maxes = {
        k: max(v, key=lambda x: x[1])
        for k, v in vals.items()
    }

    return (
        { k: v[0] for k, v in maxes.items() },
        { k: v[1] for k, v in maxes.items() }
    )


def apply_policy(model, scheduler):
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
            state_labeling = model.labeling,
            reward_models = state_rewards_from_policy(model, scheduler)
        )
    )
    

def value_iter(model, rewards, gamma, max_iter, precision = 0.01):
    vs = {state.id: 0 for state in model.states}
    error = 1
    iter = 0

    while (error > precision):
        if iter > max_iter:
            raise Exception(f"could not converge within {max_iter} iterations")

        args, vs_next = next(model, vs, rewards, gamma)

        error = max(abs(vs_next[i] - vs[i]) for i in vs)
        vs = vs_next

        iter += 1


    return (args, vs)


def interval_value_iter(model, rewards, gamma, max_iter, precision = 0.01):
    vs = {state.id: 0 for state in model.states}
    error = 1
    iter = 0

    while (error > precision):
        if iter > max_iter:
            raise Exception(f"could not converge within {max_iter} iterations")

        args, vs_next = interval_next(model, vs, rewards, gamma)

        error = max(abs(vs_next[i] - vs[i]) for i in vs)
        vs = vs_next

        iter += 1


    return (args, vs)


# def next_with_policy(model, policy, vs, rewards, gamma):
#     maxes = { state.id : 
#         (
#             (action := state.actions[policy[state.id]]).id, 
#             (
#                 rewards[(state.id, action.id)]
#                 + gamma * sum(
#                     transition.value() * vs[transition.column]
#                     for transition in action.transitions
#                 )
#             )
#         )
#         for state in model.states
#     }

#     return (
#         { k: v[0] for k, v in maxes.items() },
#         { k: v[1] for k, v in maxes.items() }
#     )


# def value_iter_with_policy(model, rewards, policy, gamma, max_iter, precision = 0.01):
#     vs = list(0 for _ in model.states)
#     error = 1
#     iter = 0

#     while (error > precision):
#         if iter > max_iter:
#             raise Exception(f"could not converge within {max_iter} iterations")

#         args, vs_next = next_with_policy(model, policy, vs, rewards, gamma)

#         error = max(abs(vs_next[i] - vs[i]) for i in vs)
#         vs = vs_next

#         iter += 1


#     return (args, vs)
