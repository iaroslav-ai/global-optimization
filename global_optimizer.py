import random
import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from copy import deepcopy

def random_id():
    return ''.join([random.choice('abcdefghijklmnopqrstuvwxyz123456789') for i in range(10)])

### VARIABLES

class AbstractVariable():
    def __init__(self, range, name, internal=False):
        self.range = range
        self.name = name
        self.internal = internal # whether a variable is user supplied or generated automatically as eg output of some function

class RealVariable(AbstractVariable):
    pass

### NODES

class AbstractNode():

    def __init__(self):
        self.x = [] # list of all variable inputs
        self.y = None # variable of output

    def __call__(self, variables):
        # returns variable or set of variables with propagated ranges
        if isinstance(variables, list):
            self.x.extend(variables)
        else:
            self.x.append(variables)
        return None

    def propagate_constraint(self):
        # propagates the boundaries of input variables to the output variables
        raise NotImplementedError()

    def concave_overestimator(self):
        # returns lambda
        raise NotImplementedError()

    def convex_underestimator(self):
        # returns lambda
        raise NotImplementedError()

    def itself(self):
        # returns lambda
        raise NotImplementedError()

class DotNode(AbstractNode):
    def __init__(self, weights, bias):
        AbstractNode.__init__(self)
        self.w = weights
        self.b = bias

    def __call__(self, variables):
        # returns variable or set of variables with propagated ranges

        AbstractNode.__call__(self, variables)
        self.y = RealVariable(None, random_id(), True) # variable range will be defined during

        return self.y

    def propagate_constraint(self):
        weights = self.w

        M, m = 0.0, 0.0  # max, min of values of result

        for v, w in zip(self.x, weights):
            m += min([w * b for b in v.range])
            M += max([w * b for b in v.range])

        m += self.b
        M += self.b

        self.y.range = [m, M]


    def concave_overestimator(self):
        # returns lambda
        return lambda x: np.dot(self.w, x) + self.b

    def convex_underestimator(self):
        # returns lambda
        return lambda x: np.dot(self.w, x) + self.b

    def itself(self):
        # returns lambda
        return lambda x: np.dot(self.w, x) + self.b


class SquareNode(AbstractNode):
    def __init__(self):
        AbstractNode.__init__(self)

    def __call__(self, variables):
        # returns variable or set of variables with propagated ranges

        AbstractNode.__call__(self, variables)
        self.y = RealVariable(None, random_id(), True)  # variable range will be defined during propagation

        return self.y

    def propagate_constraint(self):
        input = self.x[0]
        a, b = input.range

        if 0 >= a and 0 <= b:
            m = 0.0
        else:
            m = min(a*a, b*b)

        M = max(a*a, b*b)

        self.y.range = [m, M]

    def concave_overestimator(self):
        # returns lambda
        input = self.x[0]
        a, b = input.range

        return lambda x: a*a*((b-x[0]) / (b-a)) + b*b*(1 - (b-x[0]) / (b-a))

    def convex_underestimator(self):
        # returns lambda
        return lambda x: x[0]**2

    def itself(self):
        # returns lambda
        return lambda x: x[0]**2

class MultiplyNode(AbstractNode):
    def __init__(self):
        AbstractNode.__init__(self)

    def __call__(self, variables):
        # returns variable or set of variables with propagated ranges

        AbstractNode.__call__(self, variables)
        self.y = RealVariable(None, random_id(), True)  # variable range will be defined during propagation

        return self.y

    def propagate_constraint(self):
        a = self.x[0]
        b = self.x[1]

        values = []

        for x in a.range:
            for y in b.range:
                values.append(x*y)

        self.y.range = [min(values), max(values)]

    def concave_overestimator(self):
        # returns lambda
        a, b = self.x
        # get all function values in the corners as points

        X = []
        Y = []

        for x in a.range:
            for y in b.range:
                X.append([x,y])
                Y.append(x*y)

        models = []

        # construct linear functions that describe upper convex hull
        for i in range(4):
            # all points except for the ith
            Xp = [x for j,x in enumerate(X) if not (j == i)]
            Yp = [y for j,y in enumerate(Y) if not (j == i)]

            # fit linear function there
            model = LinearRegression()
            model.fit(Xp, Yp)

            # convex hull simplex must
            if Y[i] < model.predict([X[i]])[0]:
                models.append(model)

        return lambda x: min([m.predict([x])[0] for m in models])

    def convex_underestimator(self):
        # returns lambda
        a,b = self.x
        # get all function values in the corners as points

        X = []
        Y = []

        for x in a.range:
            for y in b.range:
                X.append([x,y])
                Y.append(x*y)

        models = []

        # construct linear functions that describe upper convex hull
        for i in range(4):
            # all points except for the ith
            Xp = [x for j,x in enumerate(X) if not (j == i)]
            Yp = [y for j,y in enumerate(Y) if not (j == i)]

            # fit linear function there
            model = LinearRegression()
            model.fit(Xp, Yp)

            # lower convex hull underestimates values in convex hull
            if Y[i] > model.predict([X[i]])[0]:
                models.append(model)

        return lambda x: max([m.predict([x])[0] for m in models])

    def itself(self):
        # returns lambda
        return lambda x: x[0] * x[1]


### OPTIMIZATION SUBPROBLEM

class EqVarConstraint():
    """
    This class is used to define constraints
    """
    def __init__(self, inputs, fnc, output):
        self.ip = inputs
        self.ot = output
        self.fnc = fnc

    def __call__(self, x):
        x_selection = x[self.ip]
        a = self.fnc(x_selection)
        return a - x[self.ot]

class ConvexLeqVarConstraint():
    """
    This class is used to define convex inequality constraints
    """
    def __init__(self, inputs, fnc, output, fnc_convex=True):
        self.ip = inputs
        self.ot = output
        self.fnc = fnc
        self.fnc_convex = fnc_convex

    def __call__(self, x):
        result = self.fnc(x[self.ip]) - x[self.ot]

        result = -result # current convex optimizer supports x >= 0 constraints only

        return result if self.fnc_convex else -result


class IndexConstraint():
    """
    This is used to constrain particular variable
    """
    def __init__(self, var_idx):
        self.idx = var_idx

    def __call__(self, x):
        return -(x[self.idx]) # current convex optimizer supports x >= 0 constraints only

class OptimizationProblem():
    def __init__(self):

        self.variables = {}
        self.nodes = []
        self.objective = None

        # constraints go here
        self.eq_zero = [] # variables to impose constraint of the form x = 0
        self.leq_zero = [] # variables to impose constraint of the form x <= 0

        self.children = None
        self.parent = None

        # this contain lower bound information on subproblem
        self.lower_bound = None
        self.lower_bound_x = None

        self.upper_bound = None
        self.upper_bound_x = None

    def remember_variable(self, variable):
        self.variables[variable.name] = variable

    def set_node(self, node, x):
        y = node(x)
        self.nodes.append(node)
        self.remember_variable(y)
        return y

    def propagate_constraints(self):
        # this updates constraints of all variables
        for n in self.nodes:
            n.propagate_constraint()

    def get_lower_bound(self):
        # optimize the convex relaxation
        self.propagate_constraints()

    def get_var_bounds_indicies(self):
        # returns bouds and indicies for variables
        bounds = []
        var_idx = {}  # variable name to index

        for i, kv in enumerate(self.variables.iteritems()):
            k, v = kv
            var_idx[v.name] = i
            bounds.append(v.range)

        return bounds, var_idx

    def calculate_bound(self, upper = True):
        # optimize the actual problem
        self.propagate_constraints()

        # get variable size and bounds
        bounds, var_idxs = self.get_var_bounds_indicies()

        # initial guess: middle of all boundaries
        x0 = np.array([np.random.uniform(low=b[0], high=b[1]) for b in bounds ])

        obj_idx = var_idxs[self.objective.name]

        constraints_list = []

        # generate necessary constraints
        for n in self.nodes:
            inp_idxs = [var_idxs[v.name] for v in n.x] # get indicies of input variables
            otp_idx = var_idxs[n.y.name]

            if upper:
                # use the actual function
                constraints_list.append({
                    'type':'eq',
                    'fun':EqVarConstraint(inputs=inp_idxs, fnc=n.itself(), output=otp_idx)
                })
            else:
                # constrain function values to be greater than convex underestimator ...
                constraints_list.append({
                    'type':'ineq',
                    'fun':ConvexLeqVarConstraint(
                        inputs=inp_idxs,
                        fnc=n.convex_underestimator(),
                        output=otp_idx,
                        fnc_convex=True)
                })
                # ... and less than concave overestimator
                constraints_list.append({
                    'type':'ineq',
                    'fun':ConvexLeqVarConstraint(
                        inputs=inp_idxs,
                        fnc=n.concave_overestimator(),
                        output=otp_idx,
                        fnc_convex=False)
                })

        for v in self.eq_zero:
            constraint = IndexConstraint(var_idxs[v.name])
            constraints_list.append({
                'type': 'eq',
                'fun': constraint
            })

        for v in self.leq_zero:
            constraint = IndexConstraint(var_idxs[v.name])
            constraints_list.append({
                'type': 'ineq',
                'fun': constraint
            })

        def f(x):
            return x[obj_idx]

        sol = minimize(f, x0, bounds=bounds, constraints=constraints_list)

        x = sol.x # recover solution
        x = { v.name : x[var_idxs[v.name]] for k,v in self.variables.iteritems() if not v.internal }

        # cache the upper and lower bounds
        if upper:
            self.upper_bound = sol.fun
            self.upper_bound_x = x
        else:
            self.lower_bound = sol.fun
            self.lower_bound_x = x

        return sol.fun, x

    def split_into_subproblems(self):
        """
        Splits the problem into subproblems according to random rule.

        It is assumed that the lower bound was computed on the instance of
        optimization problem P.

        :return: array of subproblems
        """

        # split randomly
        V = {k:v for k,v in self.variables.iteritems() if not v.internal}

        S = random.choice(V.keys()) # split variable name
        Sr = V[S].range # range of values to be split

        A, B = deepcopy(self), deepcopy(self)

        A.variables[S].range = [np.mean(Sr), Sr[1]]
        B.variables[S].range = [Sr[0], np.mean(Sr)]

        self.children = [A, B]
        A.parent = self
        B.parent = self

        return [A, B]

    def propagate_lower_bound(self):
        parent = self

        while not parent is None:
            parent.lower_bound = min([ch.lower_bound for ch in parent.children])
            parent = parent.parent

    def real_variable(self, range, id):
        v = RealVariable(range, id, False)
        self.remember_variable(v)
        return v

    def weighted_sum(self, constant_weights, x, constant_bias=0.0):
        # propagate the constraint
        node = DotNode(constant_weights, constant_bias)
        return self.set_node(node, x)  # this automatically remebers the output

    def square(self, x):
        # propagate the constraint
        node = SquareNode()

        if isinstance(x, list):
            return [self.set_node(node, xv) for xv in x]  # this automatically remebers the output
        else:
            return self.set_node(node, x)

    def mul(self, x, y):
        node = MultiplyNode()
        return self.set_node(node, [x, y])

    def leq_0(self, variable):
        """
        Adds variable <= 0 constraint
        :param variable: variable to constrain
        :return: nothing
        """

        self.leq_zero.append(variable)

    def eq_0(self, variable):
        """
        Adds variable == 0 constraint
        :param variable: variable to constrain
        :return: nothing
        """

        self.eq_zero.append(variable)

    def min_objective(self, variable):
        # sets the objective variable to be minimized
        if not self.objective is None:
            raise BaseException("Objective already defined")

        self.objective = variable
        self.remember_variable(variable)

# INTERFACE OPTIMIZATION CLASS

class GlobalOptimizer():
    def __init__(self, problem, epsilon=1e-3, optimality_gap = 0.0):
        """

        :param problem: OptimizationProblem instance
        :param epsilon: Inprecision that is allowed in solver
        """

        self.root_problem = problem

        self.frontier = [None] # array of items to explore

        # this contains data about the best solution found so far
        self.best_x = None
        self.best_upper_bound = None # this is used to cut down the search space
        self.best_lower_bound = None # bounds on solution

        self.epsilon = epsilon
        self.optimality_gap = optimality_gap

    def finished(self):
        return len(self.frontier) == 0

    def initialize(self):
        # this calculates initial upper bound
        p = self.root_problem

        # calculate both bounds
        p.calculate_bound(upper=True)
        p.calculate_bound(upper=False)

        self.best_upper_bound = p.upper_bound
        self.best_x = p.upper_bound_x
        self.best_lower_bound = p.lower_bound

        self.frontier = [p] # initialize the frontier




    def iterate(self):

        if len(self.frontier) == 0:
            return

        # check for termination criterion
        """"""
        if self.best_upper_bound - self.root_problem.lower_bound <= self.optimality_gap + self.epsilon:
            self.frontier = [] # finished - no need to explore any further frontier
            return

        # for now just take the top item
        instance = self.frontier.pop()

        # cut space search if possible. adding epsilon is necessary to avoid infinite loop
        # due to numeric errors coming from convex optimizer
        if instance.lower_bound + self.epsilon >= self.best_upper_bound:
            return

        # split instance
        subs = instance.split_into_subproblems()

        # calculate lower and upper bounds
        for p in subs:
            p.calculate_bound(upper=True)

            # if there is an improvement in upper bound - remeber that
            if p.upper_bound < self.best_upper_bound:
                self.best_upper_bound = p.upper_bound
                self.best_x = p.upper_bound_x

            p.calculate_bound(upper=False)

        # propagate global lower bound
        instance.propagate_lower_bound()

        # sort the list so that added top item in the frontier has the best lower bound
        subs.sort(key= lambda p: p.lower_bound, reverse=True)

        # extend the frontier
        self.frontier.extend(subs)


    def solve(self):
        self.initialize()
        idx = 0
        while not self.finished():
            print "Iteration", idx, "lower bound:", self.root_problem.lower_bound, "upper bound:", self.best_upper_bound
            idx += 1
            self.iterate()