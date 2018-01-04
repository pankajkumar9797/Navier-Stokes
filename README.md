# Navier-Stokes equation

Navier–Stokes equations are useful because they describe the physics of many phenomena of 
scientific and engineering interest. They may be used to model the weather, ocean currents, 
water flow in a pipe and air flow around a wing. The Navier–Stokes equations in their full and
simplified forms help with the design of aircraft and cars, the study of blood flow, the design
of power stations, the analysis of pollution, and many other things. 

Code is based on deal.II, an open source C++ software library supporting finite element 
code and it is interfaced with OpenBlas, P4est and Trilinos. Adaptive mesh refinement and is used 
and it is based on error estimators which enables refinement and coarsening of cells whenever 
there is a large change in the solutions gradient. This makes the code very useful as it 
doesn not need to run wherever the solution is constant, increases the mesh resolution where 
it is necessary and thereby saving computational time. visualization of solution is done 
with the help of VisIt. VisIt is an open source software developed by Lawrence Livermore 
National Laboratory and it is used for visualization, animation and as a analysis tool.

Problem is a square cavity with a flow on the upper side.

## 1. Velocity magnitude plot at a time = 0.01 seconds
![n_stokes](https://user-images.githubusercontent.com/18352934/34586656-29d1d608-f1a5-11e7-92b4-c7f6b9136e98.png)

Navier Stokes is solved by using the Chorin's projection numerical method. In this method N-S equation
is divided into 3 parts-

1. Burgers equation for calculating the velocity field without pressure term.
2. Pressure equation is solved for Neumann boundary condition.
3. Velocity is updated with the pressure calculated in the previous equation.
 

In the code the global matrices and right hand sides are assembled by using a stable scheme, 
thereafter system is solved by using either Generalized minimal residual method( in
short GMRES) solver or CG solver depending upon whether matrix is symmetric positive
definite or not.


