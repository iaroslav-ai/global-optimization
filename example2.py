from global_optimizer import OptimizationProblem, GlobalOptimizer

p = OptimizationProblem()

x = p.real_variable([-0.2, 1.0], 'x')

f = p.square(x)

f = p.weighted_sum([1.0], [f], -1.0)

f = p.square(f)

f = p.weighted_sum([1.0, 1.0], [f, x])

p.min_objective(f)

g = GlobalOptimizer(p)

g.solve()

print g.best_upper_bound, g.best_x