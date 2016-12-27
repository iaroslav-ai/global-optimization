"""
Optimize over function of the form
f(x) = x*y
x \in [-0.7, 1.0]
y \in [-0.3, 1.0]
x+y-0.7 <= 0

This is a non - convex optimization problem, that has two local minima, and its best solution is x = - 0.7, y = 1.0.
"""

from global_optimizer import OptimizationProblem, GlobalOptimizer

p = OptimizationProblem()

# create variables to be optimized over
x = p.real_variable([-0.7, 1.0], 'x')
y = p.real_variable([-0.3, 1.0], 'y')

# define computational graph - operations on variables
f = p.mul(x,y) # multiplication

# variable for inequality constraint
i = p.weighted_sum([1.0, 1.0], [x, y], -0.7)
# define constraint x + y - 0.7 = i <= 0
p.leq_0(i)

# define variable to minimize over
p.min_objective(f)

# create optimizer with description of optimization problem
g = GlobalOptimizer(p)

# solve the optimization problem
g.solve()

# print the best found objective, and minimizing set of variable values
print g.min_objective, g.min_x