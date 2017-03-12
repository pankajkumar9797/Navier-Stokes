# Burgers-equation

Burgers' equation is a fundamental partial differential equation occurring in various areas of applied 
mathematics, such as fluid mechanics, nonlinear acoustics, gas dynamics, traffic flow. It is named
for Johannes Martinus Burgers (1895–1981). 

Code is based on deal.II, an open source C++ software library supporting finite element 
code and it is interfaced with OpenBlas, P4est and Trilinos. Adaptive mesh refinement and is used 
and it is based on error estimators which enables refinement and coarsening of cells whenever 
there is a large change in the solutions gradient. This makes the code very useful as it 
doesn not need to run wherever the solution is constant, increases the mesh resolution where 
it is necessary and thereby saving computational time. visualization of solution is done 
with the help of VisIt. VisIt is an open source software developed by Lawrence Livermore 
National Laboratory and it is used for visualization, animation and as a analysis tool.

Problem is a square cavity in which a periodic source is used a right hand side function.

## 1. Velocity magnitude plot at a time = 0.01 seconds
![alt tag](https://rawgit.com/pankajkumar9797/Burgers-equation/master/doc/pics_and_videos/visit0105.png)

In the code the global matrices and right hand sides are assembled by using a stable scheme, 
thereafter system is solved by using either Generalized minimal residual method( in
short GMRES) solver or CG solver depending upon whether matrix is symmetric positive
definite or not.


## 1. L2 Error plot for a steady state with time. 
![alt tag](https://rawgit.com/pankajkumar9797/Burgers-equation/master/plot/L2_error_time.png)

In this a manufactured solution is used. Velocity in both direction is taken same.
