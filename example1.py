from global_optimizer import OptimizationProblem, GlobalOptimizer

p = OptimizationProblem()

x = p.real_variable([-1.0, 1.0], 'x')
y = p.real_variable([-1.0, 1.0], 'y')

xs = p.square(x)

f = p.weighted_sum([-1.0, 0.5, 1.0], [xs, x, y])

p.min_objective(f)

g = GlobalOptimizer(p)

g.solve()

print g.best_upper_bound, g.best_x