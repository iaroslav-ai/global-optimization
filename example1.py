from global_optimizer import OptimizationProblem, GlobalOptimizer

p = OptimizationProblem()

# create two real variables x and y as variables to be optimized over
x = p.real_variable([-1.0, 1.0], 'x')
y = p.real_variable([-1.0, 1.0], 'y')

# below the computational graph is constructed.
# square x
xs = p.square(x)

# sum variables xs, x, y
f = p.weighted_sum([-1.0, 0.5, 1.0], [xs, x, y])

# define the variable to be minimized  f
p.min_objective(f)

# initialize optimizer
g = GlobalOptimizer(p)

# solve the optimization problem
g.solve()

# print best found objective and solution
print g.min_objective, g.min_x