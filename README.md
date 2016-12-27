# Global optimization module in Python

This repository contains code that can be used to obtain solutions of
arbitrary quality to the "white box" optimization problems, that is,
 when the expression of the function being optimized is known.

A typical use case envisioned for this code is when you come up with an accurate model of some real
phenomenon, for example a model of fuel consumption depending on engine
 configuration, and then you want to find the best optimizing configuration for
 the model, for example best possible configuration that optimizes fuel consumption according
  to you model.

The code in this repo can provably find such best possible solution. Moreover,
 for suboptimal solutions an estimate is given of how much better best possible
 solution could be. For example, if during execution of engine optimization model
 you find a configuration that consumes 10 liters per 100 km, the algoritm will
 also provide a lower bound, which could be for example 9.5 liters. This means
  that no other solution would yield consumption less than 9.5 liters. Depending on
  your requirements, you might as well stop the further optimization as current solution
  (10 l per 100 km) is sufficiently "optimal".

## General description

The code finds the global optimum of optimization problem of the following form:

<p align="center">
<img src="https://github.com/iaroslav-ai/global-optimization/blob/master/images/main_1.jpg?raw=true" alt="Generic optimization problem" height="80px" />
</p>

where the objective F(x) and vector valued g(x) and h(x) are known (white box) and need not be neither linear nor convex.

The following equivalent formulation is used, that is obtained by introducing extra variables:

<p align="center">
<img src="https://github.com/iaroslav-ai/global-optimization/blob/master/images/main_2.jpg?raw=true" alt="Generic optimization problem" height="145px"/>
</p>

where f<sub>k</sub>(x) for any k ∈ K is some function, for which lower convex
underestimator function and upper concave overestimator function are available.

Currently user can select from a set of supported functions. Alternatively,
for functions of small number of arguments (up to 3) such over-
and underestimators can be estimated numerically (*coming soon*).
For functions of large number of arguments, one can try decomposing
them into computational graph with smaller number of arguments.

## Example usage

Assume you would like to solve the following optimization problem:

<p align="center">
<img src="https://github.com/iaroslav-ai/global-optimization/blob/master/images/example.jpg?raw=true" alt="Generic optimization problem" height="145px"/>
</p>

Then the code could be used as follows:

```
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
```

For more examples, see example[n ∈ {1,2,3}].py.

## Supported non - linear functions:

The functions applied to variables are defined using the instance
of `OptimizationProblem` class. Currently, the functions that are supported are:

* Weighted sum of variables, `weighted_sum`
* Square of a variable, `square`
* Multiplication of variables, `mul`
* *Coming soon: arbitrary function of up to 3 arguments*, `fnc`

Also, user can define equalities and inequalities:

* Equal zero equality, `eq_0`
* Less or equal zero, `leq_0`

## How it works

In short, this repository uses branch and bound search, where global lower bound
on solutions in some search space region is obtained using a convex relaxation
of original non - convex optimization problem in this region.
 The smaller the region, the tighter such convex relaxation become.

 The only non - convex element in the optimizatin formulation is equality
 of the form f(x) = x<sub>k</sub>. Assume for simplicity that such function f takes
 internally only a single input from vector x. Then the equality can be
 plotted as a line in 2d space:

<p align="center">
<img src="https://github.com/iaroslav-ai/global-optimization/blob/master/images/eq_relaxation.jpg?raw=true" alt="Example convex relaxation" width="500px"/>
</p>


 Such equality can be relaxed by constraining x<sub>k</sub> to belong to the convex
 set, that contains all points f<sub>k</sub>(x) = x_k. As such set is convex, the resulting
 relaxed constraint is convex.

 Such convex constraint can be defined as follows. Let o<sub>k</sub>(x) be some upper concave overestimator function for f<sub>k</sub>(x) on interval
 a<sub>k</sub> to b<sub>k</sub>, and u<sub>k</sub>(x) be the convex underestimator of f<sub>k</sub>(x) on [a<sub>k</sub>, b<sub>k</sub>]
   interval. Then the convex constraint described above can be expressed as two
   convex inequalities:

   x<sub>k</sub> - o<sub>k</sub>(x) <= 0

   u<sub>k</sub>(x) - x<sub>k</sub> <= 0

  Thus, in order to add a new function to the set of supported functions
  by this optimizer, one either needs to provide convex under and over estimators
  or such estimators should be computed numerically.

  *IMPORTANT* Both under and overestimators should become tighter
  and converge to the actual function values as the interval [a<sub>k</sub>, b<sub>k</sub>]
  becomes infinitely smaller. If this criterion is not met, then the optimization
  might not converge.
