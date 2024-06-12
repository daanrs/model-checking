from operator import itemgetter
import stormpy

class X:
    def __init__(self, id, value, lower, upper):
        self.id = id
        self.value = value
        self.lower = lower
        self.upper = upper

    def __lt__(self, other):
        return self.value < other.value

    def lower(self):
        return lower

    def upper(self):
        return upper

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
            value = vs[transition.column]
            lower = transition.value().lower()
            upper = transition.value().upper()
        )
        for transition in action.transitions
    ]

    return min_distribution(vals)
        

def min_distribution(vs):
    i = 1

    vs = list(sorted(vs))
    limit = sum(v.lower() for v in vs)

    while limit - vs[i].lower() + vs[i].upper() < 1:
        limit = limit - vs[i].lower() + vs[i].upper()
        vs[i].is_upper()
        i += 1

    vs[i].set_prob(1 - (limit - vs[i].lower()))

    for k in range(i + 1, len(vs)):
        vs[k].is_lower()

    return {v.id: v.prob for v in vs}


def next(model, vs, rewards, gamma):
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
    

def value_iter(model, precision, gamma, max_iter):
    vs = list(0 for _ in model.states)
    rewards = rewards_from_model(model)
    error = 1
    iter = 0

    while (error > precision):
        if iter > max_iter:
            raise Exception(f"could not converse within {max_iter} iterations")

        arg_and_vs = next(model, vs, rewards, gamma)

        args = list(a[0] for a in arg_and_vs)
        vs = list(a[1] for a in arg_and_vs)

        iter += 1


    return (args, vs)
