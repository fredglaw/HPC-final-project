// test script to test parareal with backward Euler
// g++-9 -std=c++11 -O3 -o be-test be-test.cpp && ./be-test
#include <stdio.h>
#include <math.h>
#include <omp.h>


double sinRHS (double t, double x){
  //RHS which just gives sin
  return -1*sin(t);
}

double fixPointSolver(double init, double (*f) (double)){
// does a fixed point iteration using f
  double tol = 1e-10; //hard-coded tolerance
  double x = init; //initialization
  double curr_f = f(x); //get first function value
  double init_resid = std::abs(x - curr_f); // initial residual
  double curr_resid = 0; //current residual
  double k = 0; // counter

  //run look while we have done < 1000 iterations, until we reduce residual by tol
  while ((curr_resid / init_resid) > tol && k < 1e3){
    x = curr_f; //save x <-- f(x)
    curr_f = f(x); //eval new f(x)
    curr_resid = std::abs(x-curr_f); //compute the norm
    k+=1;
  }
  return x; //return the fixed point
}

double backEuler(double (*righthand) (double,double), double finalTime,
               double initTime, double dt, double initx){
    /* Backward Euler solver, uses a fixed point solver to solve the algebraic
       system (hard-coded in). */
    long M = round((finalTime - initTime) / dt); //number steps to integrate
    double tol = 1e-6; //tolerance
    double u = initx; //our solution u
    double t = initTime; //initial time

    double uFP; //u for the fixed point method
    double f_uFP; //f(u) for the fixed point method, for BE this is u+dt*RHS(t+dt,u)
    double init_resid; double curr_resid; //residuals for the fixed point method
    long k; //counter for the FP

    //do all your time steps
    for (long i=1; i<= M; i++){

      //initialize the FP
      uFP = u; //initialize the FP as u
      f_uFP = u + (dt*(righthand(t+dt,uFP))); //get initial value
      init_resid = std::abs(uFP - f_uFP); // get initial residual
      curr_resid = init_resid;
      k=0;

      //run the FP
      while ((curr_resid/init_resid)>tol && k < 1000){
        uFP = f_uFP; //set u to be f(u)
        f_uFP = u + (dt*(righthand(t+dt,uFP))); //get new f(u)

        curr_resid = std::abs(uFP - f_uFP); //update residual
        k++; //increment k
      }

      //FP done, move to next step
      u = uFP; //write back to u
      t += dt; //increment t
    }
    // printf("initial value was: %f\n",initx);
    // printf("dt is: %6.2e\n",dt);
    return u;
}


int main(){
  double T = 12; // terminal time
  double M = 1e7; //ratio of course to fine
  double dt_coarse = 1; //coarse time step
  double dt_fine = dt_coarse/M; // fine time step

  long N_coarse = round(T/dt_coarse); //number of coarse time steps

  // where I save the solution using the coarse solver and the fine solver
  double* u_coarse = (double*) malloc((N_coarse+1) * sizeof(double));
  double* u_fine = (double*) malloc((N_coarse+1) * sizeof(double));
  double* truth = (double*) malloc((N_coarse+1) * sizeof(double));
  for (long i=0; i< N_coarse+1; i++){
    u_fine[i] = 0;
    u_coarse[i] = 0;
    truth[i] = 0;
  }

  // solve for a -sin RHS and set initial conditions
  double (*myRHS) (double,double) = sinRHS;
  u_coarse[0] = 1; u_fine[0] = 1;

  //get truth
  for (long i=0; i<= N_coarse; i++) truth[i] = cos((double)i*dt_coarse);

  //time coarse solve
  double timer_coarse = omp_get_wtime();
  for (long i=0; i< N_coarse; i++){
    u_coarse[i+1] = backEuler(myRHS,(double)(i+1)*dt_coarse,(double)i*dt_coarse,
                              dt_coarse,u_coarse[i]);
  }
  timer_coarse = omp_get_wtime() - timer_coarse;

  //compute coarse solve error;
  double coarse_err = 0;
  for(long i=1; i <= N_coarse; i++){
    coarse_err = fmax(std::abs(truth[i] - u_coarse[i]), coarse_err);
  }
  printf("coarse solve using dt=%2.2e took: %f s\n",dt_coarse,timer_coarse);
  printf("total coarse solver error is: %6.4e \n", coarse_err);


  //time fine solve
  double timer_fine = omp_get_wtime();
  for (long i=0; i< N_coarse; i++){
    u_fine[i+1] = backEuler(myRHS,(double)(i+1)*dt_coarse,(double)i*dt_coarse,
                              dt_fine,u_fine[i]);
  }
  timer_fine = omp_get_wtime() - timer_fine;

  //compute fine solve error
  double fine_err = 0;
  for(long i=1; i <= N_coarse; i++){
    fine_err = fmax(std::abs(truth[i] - u_fine[i]), fine_err);
  }

  printf("fine solve using dt=%2.2e took: %f s\n",dt_fine,timer_fine);
  printf("total fine solver error is: %6.4e \n", fine_err);


  //solve and time
  // double t_timer = omp_get_wtime();
  // u_fine = backEuler(myRHS,T,0,dt/M,1);
  // t_timer = omp_get_wtime() - t_timer;
  // printf("error at terminal time T=%6.2f is %6.3e\n",T,std::abs(truth-u));
  // printf("time elapse: %f s\n",t_timer);
  // printf("Final solution is u=%f and T=%f\n",u,T);

  free(u_coarse);
  free(u_fine);
  free(truth);

  return 0;
}
