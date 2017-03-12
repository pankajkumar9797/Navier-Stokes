/* ---------------------------------------------------------------------
 *
 * Copyright (C) 1999 - 2013 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Pankaj Kumar, MSc 2017.
 */

#include <deal.II/base/utilities.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/compressed_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/lac/trilinos_precondition.h>

#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>
#include <cmath>
#include <sstream>
#include <deal.II/base/logstream.h>



using namespace dealii;

template <int dim>
class N_Stokes
{
public:
  N_Stokes ();
  ~N_Stokes();
  void run ();

private:
  void make_grid ();
  void setup_system();
  void assemble_system_velocity ();
  void assemble_system_pressure ();
  void solve_velocity_system ();
  void solve_pressure_system ();
  void update_velocity ();
  void solve_update_velocity_system ();
  void refine_grid (const unsigned int min_grid_level, const unsigned int max_grid_level);
  void output_results () const;

  Triangulation<dim>   triangulation;

  FESystem<dim>        fe_velocity;
  FE_Q<dim>            fe_pressure;

  DoFHandler<dim>      dof_handler_velocity;
  DoFHandler<dim>      dof_handler_pressure;

  ConstraintMatrix     constraints_velocity;
  ConstraintMatrix     constraints_pressure;

  SparsityPattern      sparsity_pattern_velocity;
  SparsityPattern      sparsity_pattern_pressure;

  SparseILU<double>    prec_pres_Laplace;
  SparseILU<double>    prec_vel_mass;

  SparseMatrix<double> velocity_matrix;
  SparseMatrix<double> pressure_matrix;
  SparseMatrix<double> velocity_update_matrix;

  Vector<double>       old_velocity;
  Vector<double>       old_old_velocity;
  Vector<double>       velocity_solution;
  Vector<double>       update_velocity_solution;
  Vector<double>       velocity_system_rhs;
  Vector<double>       velocity_update_rhs;

  Vector<double>       old_pressure;
  Vector<double>       pressure_solution;
  Vector<double>       pressure_system_rhs;

  unsigned int         timestep_number;
  double               time_step;
  double               time;

  double               theta_imex;
  double               theta_skew;

  const double         nu = 0.001;
};


template <int dim>
class BoundaryValues : public Function<dim>
{
public:
  BoundaryValues() : Function<dim>(dim) {}
  virtual double value(const Point<dim> &p,
					 const unsigned int component) const;
  virtual void   vector_value(const Point <dim>    &p,
							Vector<double> &values) const;
};
template <int dim>
double BoundaryValues<dim>::value(const Point<dim> &p,
								const unsigned int component) const
{
  Assert (component < this->n_components,
		ExcIndexRange (component, 0, this->n_components));
  if (component == 0 && std::abs(p[dim-1]-1.0) < 1e-10)
    return 1.0;
  return 0;
}
template <int dim>
void BoundaryValues<dim>::vector_value ( const Point<dim> &p,
									   Vector<double>   &values ) const
{
  for (unsigned int c = 0; c < this->n_components; ++c)
    values(c) = BoundaryValues<dim>::value (p, c);
}


template<int dim>
class RightHandSide : public Function<dim>
{
public:
  RightHandSide (const double& time)
    :
    Function<dim>(dim),
    period (0.2),
    time(time)
  {}
  virtual double value (const Point<dim> &p,
                        const unsigned int component = 0) const;
  virtual void vector_value (const Point<dim>  &points,
		  Vector<double> &value) const;
private:
  const double period;
  const double time ;
};

template<int dim>
double RightHandSide<dim>::value (const Point<dim> &p,
                                  const unsigned int component) const
{

  Assert (dim == 2, ExcNotImplemented());

//  const double time = this->get_time();
  const double point_within_period = (time/period - std::floor(time/period));


  switch(component){
  	  case 0:
		  if ((point_within_period >= 0.0) && (point_within_period <= 0.2))
			{
			  if ((p[0] > 0.5) && (p[1] > -0.5))
				return 1;
			  else
				return 0;
			}
		  else if ((point_within_period >= 0.5) && (point_within_period <= 0.7))
			{
			  if ((p[0] > -0.5) && (p[1] > 0.5))
				return 1;
			  else
				return 0;
			}
		  else
			return 0;

  	  case 1:
		  if ((point_within_period >= 0.2) && (point_within_period <= 0.4))
			{
			  if ((p[0] > 0.5) && (p[1] > -0.5))
				return 1;
			  else
				return 0;
			}
		  else if ((point_within_period >= 0.7) && (point_within_period <= 0.9))
			{
			  if ((p[0] > -0.5) && (p[1] > 0.5))
				return 1;
			  else
				return 0;
			}
		  else
			return 0;

      default: return 0;
  }

}

template <int dim>
void
RightHandSide<dim>::vector_value (const Point<dim>  &points,
		 	 	 	 	 	 	 Vector<double> &values) const
{

	for (unsigned int c=0; c<this->n_components; ++c)
	  values(c) = RightHandSide<dim>::value (points, c);

}

template <int dim>
void right_hand_side (const std::vector<Point<dim> > &points,
                      std::vector<Tensor<1, dim> >   &values)
{
  Assert (values.size() == points.size(),
          ExcDimensionMismatch (values.size(), points.size()));
  Assert (dim >= 2, ExcNotImplemented());
  Point<dim> point_1, point_2;
  point_1(0) = 0.5;
  point_2(0) = -0.5;
  for (unsigned int point_n = 0; point_n < points.size(); ++point_n)
    {
      if (((points[point_n]-point_1).norm_square() < 0.2*0.2) ||
          ((points[point_n]-point_2).norm_square() < 0.2*0.2))
        values[point_n][0] = 1.0;
      else
        values[point_n][0] = 0.0;
      if (points[point_n].norm_square() < 0.2*0.2)
        values[point_n][1] = 1.0;
      else
        values[point_n][1] = 0.0;
    }
}

/*

template<int dim>
class RightHandSide1 : public Function<dim>
{
public:
  RightHandSide1 (const double& time) : Function<dim>(), time(time) {}
  virtual double value (const Point<dim> &p,
                        const unsigned int component = 0) const;
  const double time;

};

template<int dim>
double RightHandSide1<dim> ::value(const Point<dim> &p,
		                    const unsigned int component) const{

	return std::exp(-time)*(5 - 3*p[0]*p[0] -3*p[1]*p[1] + p[0]*p[0]*p[1]*p[1]);
}
*/
//////////////////////////

template <int dim>
class RightHandSide1 : public Function<dim>
{
public:
  RightHandSide1 () : Function<dim>(dim) {}
  virtual double value (const Point<dim>   &p,
						const unsigned int  component = 0) const;
  virtual void vector_value (const Point<dim> &p,
							 Vector<double>   &value) const;
};
template <int dim>
double
RightHandSide1<dim>::value (const Point<dim>  &p  ,
						   const unsigned int /*component */) const
{
  return 0;
}
template <int dim>
void
RightHandSide1<dim>::vector_value (const Point<dim> &p,
								  Vector<double>   &values) const
{
  for (unsigned int c=0; c<this->n_components; ++c)
	values(c) = RightHandSide1<dim>::value (p, c);
}


template <int dim>
class BubbleGauss : public dealii::Function<dim> {
public:
  BubbleGauss(const double& amplitude = 1, const double& sigma = 5, const Point<dim>& center = Point<dim>());
  double amplitude;
  double sigma;
  Point<dim> center;
  virtual double value(const Point<dim>& p, const unsigned component = 0) const;
  virtual void vector_value (const Point<dim>  &points,
		  Vector<double> &value) const;


};

template<int dim>
BubbleGauss<dim>::BubbleGauss(const double& amplitude, const double& sigma, const Point<dim>& center)  :  Function<dim>(dim), amplitude(amplitude), sigma(sigma), center(center) {

}

template<int dim>
double BubbleGauss<dim>::value(const Point<dim>& p, const unsigned component ) const {

	const double& r2 = (p - center).norm_square();

    switch (component) {
      case 0:  return  2*(p[0]*p[0] - 1)*(p[1]*p[1] - 1)*(p[0]*(p[1]*p[1] - 1) + p[1]*(p[0]*p[0] - 1))
    		            -1.0*(2.0*(p[1]*p[1] - 1) + 2*(p[0]*p[0] - 1)) ;
                                   break;//amplitude * std::exp(-sigma * r2); break;

      case 1:  return  2*(p[0]*p[0] - 1)*(p[1]*p[1] - 1)*(p[0]*(p[1]*p[1] - 1) + p[1]*(p[0]*p[0] - 1))
	                    -1.0*(2.0*(p[1]*p[1] - 1) + 2*(p[0]*p[0] - 1)) ;
                                   break;//0.1; break;
      default: return 0;
    }

}

template <int dim>
void
BubbleGauss<dim>::vector_value (const Point<dim>  &points,
		 	 	 	 	 	 	 Vector<double> &values) const
{

	for (unsigned int c=0; c<this->n_components; ++c)
	  values(c) = BubbleGauss<dim>::value (points, c);

}

template <int dim>
class ExactSolution : public dealii::Function<dim> {
public:
  ExactSolution(): Function<dim>(dim){} ;
  virtual void vector_value (const Point<dim>  &points,
		  Vector<double> &value) const;

};


template <int dim>
void
ExactSolution<dim>::vector_value (const Point<dim>  &p,
		 	 	 	 	 	 	 Vector<double> &values) const
{
    Assert (values.size() == dim,
            ExcDimensionMismatch (values.size(), dim));

    values(0) = (p[0]*p[0] - 1)*(p[1]*p[1] - 1) ;
    values(1) = (p[0]*p[0] - 1)*(p[1]*p[1] - 1) ;

}


template <int dim>
N_Stokes<dim>::N_Stokes ()
  :
  fe_velocity (FE_Q<dim>(2), dim),
  fe_pressure(1),
  dof_handler_velocity (triangulation),
  dof_handler_pressure (triangulation),
  timestep_number(0),
  time_step(1. / 500),
  time(0),
  theta_imex(0.5),
  theta_skew(0.5)
{}

template <int dim>
N_Stokes<dim>::~N_Stokes (){
  dof_handler_velocity.clear ();
  dof_handler_pressure.clear ();
}

template <int dim>
void N_Stokes<dim>::make_grid ()
{
  GridGenerator::hyper_cube(triangulation);
//  GridGenerator::hyper_cube (triangulation, -1, 1);
  triangulation.refine_global (5);

  std::cout << "   Number of active cells: "
            << triangulation.n_active_cells()
            << std::endl
            << "   Total number of cells: "
            << triangulation.n_cells()
            << std::endl;
}


template <int dim>
void N_Stokes<dim>::setup_system ()
{
  dof_handler_velocity.distribute_dofs (fe_velocity);
  dof_handler_pressure.distribute_dofs (fe_pressure);

  std::cout << "   Number of degrees of freedom(velocity + pressure): "
            << dof_handler_velocity.n_dofs() + dof_handler_pressure.n_dofs()
            << std::endl;



  {
  constraints_velocity.clear ();

  DoFTools::make_hanging_node_constraints(dof_handler_velocity, constraints_velocity);

  VectorTools::interpolate_boundary_values (dof_handler_velocity,
                                            0,
                                            BoundaryValues<dim>(),
                                            constraints_velocity);

  std::set<types::boundary_id> no_normal_flux_boundaries;
  no_normal_flux_boundaries.insert (0);
  VectorTools::compute_no_normal_flux_constraints (dof_handler_velocity, 0,
                                                   no_normal_flux_boundaries,
                                                   constraints_velocity);

  }
  constraints_velocity.close();


  constraints_pressure.clear ();

  DoFTools::make_hanging_node_constraints(dof_handler_pressure, constraints_pressure);

  VectorTools::interpolate_boundary_values (dof_handler_pressure,
                                            0,
                                            ZeroFunction<dim>(),
                                            constraints_pressure);


  constraints_pressure.close();

  DynamicSparsityPattern  velocity_sparsity(dof_handler_velocity.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler_velocity, velocity_sparsity, constraints_velocity, /*keep_constrained_dofs = */ true);

  sparsity_pattern_velocity.copy_from(velocity_sparsity);

  DynamicSparsityPattern  pressure_sparsity(dof_handler_pressure.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler_pressure, pressure_sparsity, constraints_pressure, /*keep_constrained_dofs = */ true);

  sparsity_pattern_pressure.copy_from(pressure_sparsity);

  velocity_matrix.reinit (sparsity_pattern_velocity);
  pressure_matrix.reinit (sparsity_pattern_pressure);
  velocity_update_matrix.reinit (sparsity_pattern_velocity);

  old_velocity.reinit(dof_handler_velocity.n_dofs());
  old_old_velocity.reinit(dof_handler_velocity.n_dofs());
  velocity_solution.reinit (dof_handler_velocity.n_dofs());
  update_velocity_solution.reinit (dof_handler_velocity.n_dofs());

  velocity_system_rhs.reinit (dof_handler_velocity.n_dofs());
  velocity_update_rhs.reinit (dof_handler_velocity.n_dofs());

  old_pressure.reinit(dof_handler_pressure.n_dofs());
  pressure_solution.reinit (dof_handler_pressure.n_dofs());
  pressure_system_rhs.reinit (dof_handler_pressure.n_dofs());

}


template <int dim>
void N_Stokes<dim>::assemble_system_velocity ()
{
  QGauss<dim>  quadrature_formula(2);

//  const BubbleGauss<dim>  right_hand_side;
//  const RightHandSide<dim> right_hand_side(time);
    const ZeroFunction<dim>   right_hand_side(dim);

  velocity_matrix = 0;
  velocity_system_rhs    = 0;

  FEValues<dim> fe_values (fe_velocity, quadrature_formula,
                           update_values   | update_gradients |
                           update_quadrature_points | update_JxW_values);


  const unsigned int   dofs_per_cell = fe_velocity.dofs_per_cell;
  const unsigned int   n_q_points    = quadrature_formula.size();

  FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
  Vector<double>       cell_rhs (dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
  std::vector<Tensor<1, dim> >                 old_values (n_q_points);
  std::vector<Tensor<1, dim> >                 old_old_values (n_q_points);
  std::vector<Tensor<2, dim> >                 old_grad (n_q_points);
  std::vector<double>                          old_div(n_q_points);
  std::vector<double>                          old_old_div(n_q_points);

  std::vector<Vector<double> >      rhs_values (n_q_points, Vector<double>(dim));

  typename DoFHandler<dim>::active_cell_iterator
  cell = dof_handler_velocity.begin_active(),
  endc = dof_handler_velocity.end();

  for (; cell!=endc; ++cell)
    {
      fe_values.reinit (cell);

      const FEValuesViews::Vector<dim>& fe_vector_values = fe_values[FEValuesExtractors::Vector(0)];

      cell_matrix = 0;
      cell_rhs = 0;
      fe_vector_values.get_function_values (old_velocity, old_values);
      fe_vector_values.get_function_gradients(old_velocity, old_grad);
      fe_vector_values.get_function_divergences(old_velocity, old_div);

      fe_vector_values.get_function_values (old_old_velocity, old_old_values);
      fe_vector_values.get_function_divergences(old_old_velocity, old_old_div);


      right_hand_side.vector_value_list(fe_values.get_quadrature_points(),
                                  rhs_values);


      for (unsigned int q_index=0; q_index<n_q_points; ++q_index){


          Tensor<1, dim> rhs_val;

          for (int d = 0; d < dim; ++d) {
            rhs_val[d] += right_hand_side.value(fe_values.quadrature_point(q_index), d);
          }

    	  const double& u_star_div = old_div[q_index]; //2*old_div[q_index]    - old_old_div[q_index] ;
    	  const Tensor<1, dim>& u_star     = old_values[q_index]; //2*old_values[q_index] - old_old_values[q_index];

        for (unsigned int i=0; i<dofs_per_cell; ++i)
          {
            const Tensor<1, dim>& u_val   = fe_vector_values.value(i, q_index);
            const Tensor<2, dim>& u_grad  = fe_vector_values.gradient(i, q_index);

            for (unsigned int j=0; j<dofs_per_cell; ++j) {

                const Tensor<1, dim>& v_val   = fe_vector_values.value(j, q_index);
                const Tensor<2, dim>& v_grad  = fe_vector_values.gradient(j, q_index);

			    cell_matrix(i,j) += ( u_val * v_val //0.5*(3*u_val * v_val)
								     +
								    time_step*contract3(u_star, u_grad, v_val)//u_star*u_grad*v_val
								     +
								    0.5*time_step*u_star_div*contract(u_val, v_val)
								     +
								    nu*time_step*double_contract(u_grad, v_grad)
									   )*fe_values.JxW (q_index);
            }
            				//2.0*old_values[q_index]* u_val - 0.5*old_old_values[q_index]* u_val
            cell_rhs(i) += (old_values[q_index]* u_val  + time_step * (rhs_val * u_val)
                               )* fe_values.JxW (q_index);
          }
      }
      cell->get_dof_indices (local_dof_indices);
     constraints_velocity.distribute_local_to_global(cell_matrix,
                                          cell_rhs,
                                          local_dof_indices,
                                          velocity_matrix,
                                          velocity_system_rhs);

    }


//  BoundaryValues<dim> boundary_values_function;
//  boundary_values_function.set_time(time);

  std::map<types::global_dof_index,double> boundary_values;
  VectorTools::interpolate_boundary_values (dof_handler_velocity,
                                            0,
                                            BoundaryValues<dim>(),//ZeroFunction<dim>(dim),
                                            boundary_values);
  MatrixTools::apply_boundary_values (boundary_values,
                                      velocity_matrix,
                                      velocity_solution,
                                      velocity_system_rhs);

}


template <int dim>
void N_Stokes<dim>::assemble_system_pressure ()
{
  QGauss<dim>    quadrature_formula(2);
  QGauss<dim-1>  face_quadrature_formula(4);

//  const BubbleGauss<dim>  right_hand_side;
  const RightHandSide<dim> right_hand_side(time);
//    const ZeroFunction<dim>   right_hand_side(dim);

  pressure_matrix = 0;
  pressure_system_rhs    = 0;

  FEValues<dim> fe_values (fe_pressure, quadrature_formula,
                           update_values   | update_gradients |
                           update_quadrature_points | update_JxW_values);

  FEFaceValues<dim> fe_face_values (fe_pressure, face_quadrature_formula,
                           update_values   | update_gradients |update_normal_vectors |
                           update_quadrature_points | update_JxW_values);

  FEValues<dim> fe_values_velocity (fe_velocity, quadrature_formula,
                           update_values   | update_gradients );


  FEFaceValues<dim> fe_face_values_velocity (fe_velocity, face_quadrature_formula,
                           update_values );


  const unsigned int   dofs_per_cell = fe_pressure.dofs_per_cell;
  const unsigned int   n_q_points    = quadrature_formula.size();
  const unsigned int   n_q_face      = face_quadrature_formula.size();

  FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
  Vector<double>       cell_rhs (dofs_per_cell);

  std::vector<types::global_dof_index>         local_dof_indices (dofs_per_cell);
  std::vector<double >                         velocity_div_values (n_q_points);
  std::vector<Tensor<1, dim> >                 face_velocity_values (n_q_points);

  std::vector<Vector<double> >                 rhs_values (n_q_points, Vector<double>(dim));

  const double dt_1 = 1.0/time_step;


  typename DoFHandler<dim>::active_cell_iterator
  cell = dof_handler_pressure.begin_active(),
  endc = dof_handler_pressure.end();
  typename DoFHandler<dim>::active_cell_iterator
  cell_v = dof_handler_velocity.begin_active();
  for (; cell!=endc; ++cell, ++cell_v)
    {
      fe_values.reinit (cell);
      fe_values_velocity.reinit(cell_v);

      const FEValuesViews::Vector<dim>& fe_vector_values = fe_values_velocity[FEValuesExtractors::Vector(0)];
      fe_vector_values.get_function_divergences(velocity_solution, velocity_div_values);

      cell_matrix = 0;
      cell_rhs = 0;

      for (unsigned int q_index=0; q_index<n_q_points; ++q_index){

        for (unsigned int i=0; i<dofs_per_cell; ++i)
          {
            const double & u_val   = fe_values.shape_value(i, q_index);
            const Tensor<1, dim>& u_grad  = fe_values.shape_grad(i, q_index);

            for (unsigned int j=0; j<dofs_per_cell; ++j) {

                const Tensor<1, dim>& v_grad  = fe_values.shape_grad(j, q_index);

              cell_matrix(i,j) += ( u_grad * v_grad
                                       )*fe_values.JxW (q_index);
            }

            cell_rhs(i) +=  - dt_1*(velocity_div_values[q_index] * u_val
                               )* fe_values.JxW (q_index);
          }
      }

      for (unsigned int iface = 0; iface < GeometryInfo<dim>::faces_per_cell; ++iface) {
        if (cell->face(iface)->at_boundary() == false)
          continue;
        fe_face_values.reinit(cell, iface);
        fe_face_values_velocity.reinit(cell_v, iface);

		const FEValuesViews::Vector<dim>& fe_vector_values = fe_face_values_velocity[FEValuesExtractors::Vector(0)];
		fe_vector_values.get_function_values(velocity_solution, face_velocity_values);

        for (unsigned q = 0; q < n_q_face; ++q) {
          const Tensor<1, dim>& normal  = fe_face_values.normal_vector(q);
          const double&         n_u     = normal * face_velocity_values[q];
          for (unsigned i = 0; i < dofs_per_cell; ++i) {
            const double&         q_val = fe_face_values.shape_value(i, q);
            cell_rhs(i) += (
                dt_1 * n_u * q_val
              ) * fe_face_values.JxW(q);
          }
        }

      }

      cell->get_dof_indices (local_dof_indices);
     constraints_pressure.distribute_local_to_global(cell_matrix,
                                          cell_rhs,
                                          local_dof_indices,
                                          pressure_matrix,
                                          pressure_system_rhs);


    }

  std::map<types::global_dof_index,double> boundary_values;
  VectorTools::interpolate_boundary_values (dof_handler_pressure,
                                            0,
                                            ZeroFunction<dim>(),
                                            boundary_values);
  MatrixTools::apply_boundary_values (boundary_values,
                                      pressure_matrix,
                                      pressure_solution,
                                      pressure_system_rhs);


}

template <int dim>
void N_Stokes<dim>::solve_velocity_system ()
{
/*  SolverControl           solver_control (1000, 1e-8 * system_rhs.l2_norm());
  SolverCG<>              solver (solver_control);

  PreconditionSSOR<> preconditioner;//PreconditionIdentity()
  preconditioner.initialize(system_matrix, 1.0);

  solver.solve (system_matrix, solution, system_rhs,
                preconditioner);

*/
	int    vel_max_its     = 5000;
	double vel_eps         = 1e-9;
	int    vel_Krylov_size = 30;

  PreconditionSSOR<> preconditioner;
  preconditioner.initialize(velocity_matrix, 1.0);

	SolverControl solver_control (vel_max_its, vel_eps*velocity_system_rhs.l2_norm());
	{
		SolverGMRES<Vector<double>> gmres1 (solver_control,
						   SolverGMRES<>::AdditionalData (vel_Krylov_size));
		gmres1.solve (velocity_matrix, velocity_solution, velocity_system_rhs, preconditioner);
	}



  std::cout << "   " << solver_control.last_step()
            << " GMRES iterations needed to obtain convergence."
            << std::endl;

  constraints_velocity.distribute (velocity_solution);
}


template <int dim>
void N_Stokes<dim>::solve_pressure_system ()
{
  SolverControl           solver_control (1000, 1e-9 * pressure_system_rhs.l2_norm());
  SolverCG<>              solver (solver_control);

  PreconditionSSOR<> preconditioner;//PreconditionIdentity()
  preconditioner.initialize(pressure_matrix, 1.0);


  prec_pres_Laplace.initialize(pressure_matrix,
                               SparseILU<double>::AdditionalData (0.01,
                                   60) );

  solver.solve (pressure_matrix, pressure_solution, pressure_system_rhs,
		  prec_pres_Laplace);



  std::cout << "   " << solver_control.last_step()
            << " CG iterations needed to obtain convergence."
            << std::endl;

  constraints_pressure.distribute (pressure_solution);
}

template <int dim>
void N_Stokes<dim>::update_velocity (){

	  QGauss<dim>  quadrature_formula(2);

	  velocity_update_matrix = 0;
	  velocity_update_rhs    = 0;

	  FEValues<dim> fe_values (fe_velocity, quadrature_formula,
	                           update_values   | update_gradients |
	                           update_quadrature_points | update_JxW_values);

	  FEValues<dim> fe_pressure_values (fe_pressure, quadrature_formula,
	                           update_values   | update_gradients |
	                           update_quadrature_points | update_JxW_values);


	  const unsigned int   dofs_per_cell = fe_velocity.dofs_per_cell;
	  const unsigned int   n_q_points    = quadrature_formula.size();

	  FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
	  Vector<double>       cell_rhs (dofs_per_cell);

	  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
	  std::vector<Tensor<1, dim> >                 vel_values (n_q_points);
	  std::vector<Tensor<1, dim> >                 pres_grad (n_q_points);

	  typename DoFHandler<dim>::active_cell_iterator
	  cell = dof_handler_velocity.begin_active(),
	  endc = dof_handler_velocity.end();

	  typename DoFHandler<dim>::active_cell_iterator
	  cell_p = dof_handler_pressure.begin_active();

	  for (; cell!=endc; ++cell, ++cell_p)
	    {
	      fe_values.reinit (cell);
	      fe_pressure_values.reinit (cell_p);

	      const FEValuesViews::Vector<dim>& fe_vector_values = fe_values[FEValuesExtractors::Vector(0)];

	      cell_matrix = 0;
	      cell_rhs = 0;
	      fe_vector_values.get_function_values (velocity_solution, vel_values);
	      fe_pressure_values.get_function_gradients(pressure_solution, pres_grad);


	      for (unsigned int q_index=0; q_index<n_q_points; ++q_index){

	        for (unsigned int i=0; i<dofs_per_cell; ++i)
	          {
	            const Tensor<1, dim>& u_val   = fe_vector_values.value(i, q_index);

	            for (unsigned int j=0; j<dofs_per_cell; ++j) {

	                const Tensor<1, dim>& v_val   = fe_vector_values.value(j, q_index);

				    cell_matrix(i,j) += ( u_val * v_val
										   )*fe_values.JxW (q_index);
	            }

	            cell_rhs(i) += (vel_values[q_index]* u_val  - time_step * (pres_grad[q_index] * u_val)
	                               )* fe_values.JxW (q_index);
	          }
	      }
	      cell->get_dof_indices (local_dof_indices);
	     constraints_velocity.distribute_local_to_global(cell_matrix,
	                                          cell_rhs,
	                                          local_dof_indices,
	                                          velocity_update_matrix,
	                                          velocity_update_rhs);

	    }


	//  BoundaryValues<dim> boundary_values_function;
	//  boundary_values_function.set_time(time);
/*
	  std::map<types::global_dof_index,double> boundary_values;
	  VectorTools::interpolate_boundary_values (dof_handler_velocity,
	                                            0,
	                                            ZeroFunction<dim>(dim),
	                                            boundary_values);
	  MatrixTools::apply_boundary_values (boundary_values,
	                                      velocity_update_matrix,
	                                      update_velocity_solution,
	                                      velocity_update_rhs);
*/
}

template <int dim>
void N_Stokes<dim>::solve_update_velocity_system ()
{/*
	int    vel_max_its     = 5000;
	double vel_eps         = 1e-9;
	int    vel_Krylov_size = 30;

  PreconditionSSOR<> preconditioner;
  preconditioner.initialize(velocity_update_matrix, 1.0);

	SolverControl solver_control (vel_max_its, vel_eps*velocity_update_rhs.l2_norm());
	{
		SolverGMRES<Vector<double>> gmres1 (solver_control,
						   SolverGMRES<>::AdditionalData (vel_Krylov_size));
		gmres1.solve (velocity_update_matrix, update_velocity_solution, velocity_update_rhs, preconditioner);
	}

  std::cout << "   " << solver_control.last_step()
            << " GMRES iterations needed to obtain convergence."
            << std::endl;

*/
	  SolverControl           solver_control (1000, 1e-6 * pressure_system_rhs.l2_norm());
	  SolverCG<>              solver (solver_control);

	  prec_vel_mass.initialize(velocity_update_matrix,
	                               SparseILU<double>::AdditionalData (0.01,
	                                   60) );

	  solver.solve (velocity_update_matrix, update_velocity_solution, velocity_update_rhs, prec_vel_mass);



	  std::cout << "   " << solver_control.last_step()
	            << " CG iterations needed to obtain convergence."
	            << std::endl;

  constraints_velocity.distribute (update_velocity_solution);
}


template <int dim>
void N_Stokes<dim>::refine_grid(const unsigned int min_grid_level,
		                     const unsigned int max_grid_level){

	Vector<float> estimated_error_per_cell(triangulation.n_active_cells());

	KellyErrorEstimator<dim>::estimate(dof_handler_velocity,
			                            QGauss<dim-1>(fe_velocity.degree+2),
			                            typename FunctionMap<dim>::type(),
			                            update_velocity_solution,
			                            estimated_error_per_cell);
/*
	GridRefinement::refine_and_coarsen_fixed_number(triangulation,
			                                         estimated_error_per_cell,
			                                         0.6, 0.4);
	*/

    GridRefinement::refine_and_coarsen_fixed_number (triangulation,
                                                       estimated_error_per_cell,
                                                       0.5, 0.2);

	if(triangulation.n_levels() > max_grid_level){
		for(typename Triangulation<dim>::active_cell_iterator
				cell = triangulation.begin_active(max_grid_level);
				cell != triangulation.end(); ++cell){
			cell->clear_refine_flag();
		}
	}
	for(typename Triangulation<dim>::active_cell_iterator
			cell = triangulation.begin_active(min_grid_level);
			cell != triangulation.end_active(min_grid_level); ++cell)
		cell->clear_coarsen_flag();

	SolutionTransfer<dim> solution_transfer(dof_handler_velocity);

	Vector<double> previous_solution;
	previous_solution = update_velocity_solution;


	triangulation.prepare_coarsening_and_refinement();
	solution_transfer.prepare_for_coarsening_and_refinement(previous_solution);


	triangulation.execute_coarsening_and_refinement();
	setup_system();

	solution_transfer.interpolate(previous_solution, update_velocity_solution);

	constraints_velocity.distribute(update_velocity_solution);

}



template <int dim>
void N_Stokes<dim>::output_results () const
{
    std::vector<std::string> solution_names (dim, "velocity");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation
    (dim, DataComponentInterpretation::component_is_part_of_vector);

    DataOut<dim> data_out;
    data_out.attach_dof_handler (dof_handler_velocity);
    data_out.add_data_vector (update_velocity_solution, solution_names,
                              DataOut<dim>::type_dof_data,
                              data_component_interpretation);

    data_out.add_data_vector (dof_handler_pressure, pressure_solution,
                              "Pressure");

    data_out.build_patches ();
    std::ostringstream filename;
    filename << "solution-"
             << Utilities::int_to_string (timestep_number, 4)
             << ".vtk";
    std::ofstream output (filename.str().c_str());
    data_out.write_vtk (output);

//////////////////////////////////////////////////////////////
/*    const FESystem<dim> joint_fe (fe_velocity, 1,
                                  fe_pressure, 1);
    DoFHandler<dim> joint_dof_handler (triangulation);
    joint_dof_handler.distribute_dofs (joint_fe);
    Assert (joint_dof_handler.n_dofs() ==
            (dof_handler_velocity.n_dofs() +
             dof_handler_pressure.n_dofs()),
            ExcInternalError());
    static Vector<double> joint_solution (joint_dof_handler.n_dofs());

    std::vector<types::global_dof_index> loc_joint_dof_indices (joint_fe.dofs_per_cell),
        loc_vel_dof_indices (fe_velocity.dofs_per_cell),
        loc_pres_dof_indices (fe_pressure.dofs_per_cell);
    typename DoFHandler<dim>::active_cell_iterator
    joint_cell = joint_dof_handler.begin_active(),
    joint_endc = joint_dof_handler.end(),
    vel_cell   = dof_handler_velocity.begin_active(),
    pres_cell  = dof_handler_pressure.begin_active();
    for (; joint_cell != joint_endc; ++joint_cell, ++vel_cell, ++pres_cell)
      {
        joint_cell->get_dof_indices (loc_joint_dof_indices);
        vel_cell->get_dof_indices (loc_vel_dof_indices),
                 pres_cell->get_dof_indices (loc_pres_dof_indices);
        for (unsigned int i=0; i<joint_fe.dofs_per_cell; ++i)
          if(joint_fe.system_to_base_index(i).first.first == 0)
            {
              Assert (joint_fe.system_to_base_index(i).second < loc_vel_dof_indices.size(),
                      ExcInternalError());
              joint_solution (loc_joint_dof_indices[i]) =
              velocity_solution(loc_vel_dof_indices[ joint_fe.system_to_base_index(i).second ]);
            }
          else{
              Assert (joint_fe.system_to_base_index(i).first.first == 1,
                      ExcInternalError());

              Assert (joint_fe.system_to_base_index(i).second < loc_pres_dof_indices.size(),
                      ExcInternalError());
              joint_solution (loc_joint_dof_indices[i]) =
                pressure_solution (loc_pres_dof_indices[ joint_fe.system_to_base_index(i).second ]);

            }
      }



    std::vector<std::string> joint_solution_names (dim, "v");
    joint_solution_names.push_back ("p");

    DataOut<dim> data_out;
    data_out.attach_dof_handler (joint_dof_handler);
    std::vector< DataComponentInterpretation::DataComponentInterpretation >
    component_interpretation (dim+1,
                              DataComponentInterpretation::component_is_part_of_vector);
    component_interpretation[dim]
      = DataComponentInterpretation::component_is_scalar;

    data_out.add_data_vector (joint_solution,
                              joint_solution_names,
                              DataOut<dim>::type_dof_data,
                              component_interpretation);
    data_out.build_patches ();
    std::ofstream output (("solution-" +
                           Utilities::int_to_string (step, 4) +
                           ".vtk").c_str());
    data_out.write_vtk (output);

    std::ostringstream filename;
    filename << "solution-"
             << Utilities::int_to_string (timestep_number, 4)
             << ".vtk";
    std::ofstream output (filename.str().c_str());
    data_out.write_vtk (output);
*/
}




template <int dim>
void N_Stokes<dim>::run ()
{
  std::cout << "Solving problem in " << dim << " space dimensions." << std::endl;

  make_grid();
  setup_system ();
  const BubbleGauss<dim> bubble_gum;
  unsigned int pre_refinement_step = 0;
  const unsigned int n_adaptive_pre_refinement_steps = 4;
  const unsigned int initial_global_refinement = 2;

start_time_iteration:

  timestep_number = 0;
  time            = 0;

  VectorTools::project (dof_handler_velocity,
                        constraints_velocity,
                        QGauss<dim>(2),
                        ZeroFunction<dim>(dim),//bubble_gum,
                        old_velocity);

  VectorTools::project (dof_handler_pressure,
                        constraints_pressure,
                        QGauss<dim>(2),
                        ZeroFunction<dim>(),
                        pressure_solution);

  update_velocity_solution = old_velocity;
  output_results();
//  old_solution = 0;
//  solution     = 0;

   do{

      std::cout << "Time step " << timestep_number << " at t=" << time
                << std::endl;

      assemble_system_velocity ();
      solve_velocity_system ();

      assemble_system_pressure ();
      solve_pressure_system();

      update_velocity ();
      solve_update_velocity_system ();

      output_results ();

      if((timestep_number ==0)&& (pre_refinement_step < n_adaptive_pre_refinement_steps)){

    	  refine_grid(initial_global_refinement,
    			  initial_global_refinement + n_adaptive_pre_refinement_steps);
    	  ++pre_refinement_step;
    	  old_old_velocity.reinit(update_velocity_solution.size());
    	  old_velocity.reinit(update_velocity_solution.size());
    	  velocity_system_rhs.reinit(update_velocity_solution.size());
    	  velocity_solution.reinit(update_velocity_solution.size());

    	  goto start_time_iteration;

      }
      else if ((timestep_number > 0) && (timestep_number % 5 == 0)){

    	  refine_grid(initial_global_refinement,
    			  initial_global_refinement + n_adaptive_pre_refinement_steps);
    	  old_old_velocity.reinit(update_velocity_solution.size());
     	  old_velocity.reinit(update_velocity_solution.size());
     	  velocity_system_rhs.reinit(update_velocity_solution.size());
   	      velocity_solution.reinit(update_velocity_solution.size());
      }

      time += time_step;
      ++timestep_number;

      old_old_velocity = old_velocity;
      old_velocity = update_velocity_solution;

      velocity_solution = 0;
      pressure_solution = 0;
      update_velocity_solution = 0;

  }while (time <= 1.0);


}



int main ()
{

  try
    {
      using namespace dealii;
      deallog.depth_console(0);

      N_Stokes<2> navier_stokes_equation_solver;
      navier_stokes_equation_solver.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl << exc.what()
                << std::endl << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl << "Aborting!"
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  return 0;
}
