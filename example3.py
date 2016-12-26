from global_optimizer import OptimizationProblem, GlobalOptimizer

p = OptimizationProblem()

x = p.real_variable([-0.7, 1.0], 'x')
y = p.real_variable([-0.3, 1.0], 'y')

f = p.mul(x,y)

p.min_objective(f)

g = GlobalOptimizer(p)

g.solve()

print g.best_upper_bound, g.best_x