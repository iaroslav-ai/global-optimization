"""
Optimize over function of the form
f(x) = (x^2-1)^2 + x
x \in [-0.2, 1.0]

This is a non - convex function, that has two local minima:

https://www.google.com/search?q=(x%5E2-1)%5E2+%2B+x&oq=(x%5E2-1)%5E2+%2B+x&aqs=chrome..69i57j0l5.204j0j4&client=ubuntu&sourceid=chrome&ie=UTF-8
"""

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

print g.min_objective, g.min_x