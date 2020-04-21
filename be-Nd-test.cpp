// test script to test parareal with backward Euler
// g++-9 -std=c++11 -fopenmp -O2 -o be-Nd-test be-Nd-test.cpp && ./be-Nd-test
#include <stdio.h>
#include <cmath>
#include <cstdlib>
#include <omp.h>

#define N_THREADS 2

void sinRHS (double* right, double t, double* x, long N){
  //RHS which just gives sin
  for (long k=0; k<N; k++) right[k] = -1*sin((k+1)*t);
}

void backEuler(void (*RH_handle) (double*,double,double*,long), //RHS function pointer
                 double finalTime, // T_final to integrate to
                 double initTime, // T_init to begin at
                 double dt, // time step
                 double* u, // address of solution at current time to write to.
                 double* righthand, // buffer to hold RHS solution
                 double* uFP, // buffer to hold solution for fixed-point iterations (also serves as IC)
                 double* f_uFP, // buffer to hold RHS for fixed-point iterations
                 long N //dimension of the ODE
                 ){
    /* Backward Euler solver, uses a fixed point solver to solve the algebraic
       system (hard-coded in). */
    long M = round((finalTime - initTime) / dt); //number steps to integrate
    double tol = 1e-6; //tolerance (hard-coded in)
    double t = initTime; //initial time

    double init_resid; double curr_resid; //residuals for the fixed point method
    long FPcounter; //counter for the FP

    //the fixed point buffer holds the final solution at the previous time step
    // this is the IC for this time step, write into u
    for (long j=0; j<N; j++) u[j] = uFP[j];

    //do all your time steps
    for (long k=1; k<= M; k++){

      //initialize the FP
      // for (long j=0; j<N; j++) uFP[j] = u[j]; //initialize the FP as u
      RH_handle(righthand,t+dt,uFP,N); //update the RHS
      for (long j=0; j<N; j++) f_uFP[j] = u[j] + dt*righthand[j];
      // f_uFP = u + (dt*(righthand(t+dt,uFP))); //get initial value
      init_resid = 0;
      for (long j=0; j<N; j++) init_resid += std::abs(uFP[j]-f_uFP[j]);
      // init_resid = std::abs(uFP - f_uFP); // get initial residual
      curr_resid = init_resid;
      FPcounter=0;

      //run the FP
      while ((curr_resid/init_resid)>tol && FPcounter < 1000){
        for (long j=0; j<N; j++) uFP[j] = f_uFP[j]; //set u to be f(u)

        RH_handle(righthand,t+dt,uFP,N); //update the RHS
        for (long j=0; j<N; j++) f_uFP[j] = u[j] + dt*righthand[j];

        curr_resid = 0;
        for (long j=0; j<N; j++) curr_resid += std::abs(uFP[j]-f_uFP[j]); //update residual
        FPcounter++; //increment counter
      }

      //FP done, move to next step
      for(long j=0; j<N; j++) u[j] = uFP[j]; //write back to u
      t += dt; //increment t
    }
    // printf("initial value was: %f\n",initu);
    // printf("dt is: %6.2e\n",dt);
}


int main(){
  double T = 100; // terminal time
  double M = 1e4; //ratio of course to fine
  double dt_coarse = 10; //coarse time step
  double dt_fine = dt_coarse/M; // fine time step
  long N = 50;
  long K = 2; //number of parareal iterations

  long N_coarse = round(T/dt_coarse); //number of coarse time steps

  //timeVec is the vector of times to get solution at
  double *timeVec = (double*) malloc((N_coarse+1) * sizeof(double));
  for(long s=0; s<= N_coarse; s++) timeVec[s] = (double)s*dt_coarse;

  // where I save the solution using the coarse solver and the fine solver
  double* u_coarse = (double*) malloc(N*(N_coarse+1) * sizeof(double));
  double* u_fine = (double*) malloc(N*(N_coarse+1) * sizeof(double));
  double* truth = (double*) malloc(N*(N_coarse+1) * sizeof(double));
  for (long s=0; s< N*(N_coarse+1); s++){
    u_fine[s] = 0;
    u_coarse[s] = 0;
    truth[s] = 0;
  }

  // allocate buffers for the solver
  double* coarse_righthand_buff = (double*) malloc(N*sizeof(double));
  double* coarse_FPsoln_buff = (double*) malloc(N*sizeof(double));
  double* coarse_FPrighthand_buff = (double*) malloc(N*sizeof(double));
  double* fine_righthand_buff = (double*) malloc(N*sizeof(double));
  double* fine_FPsoln_buff = (double*) malloc(N*sizeof(double));
  double* fine_FPrighthand_buff = (double*) malloc(N*sizeof(double));

  // solve for a -sin RHS and set initial conditions
  void (*myRHS) (double*,double,double*,long) = sinRHS;
  for (long n=0; n<N; n++){
    u_coarse[n] = 1./((double)n+1);
    u_fine[n] = 1./((double)n+1);
    coarse_righthand_buff[n] = 0;
    fine_righthand_buff[n] = 0;
    coarse_FPsoln_buff[n] = u_coarse[n];
    fine_FPsoln_buff[n] = u_fine[n];
    coarse_FPrighthand_buff[n] = 0;
    fine_FPrighthand_buff[n] = 0;
  }

  //get truth
  for (long j=0; j<= N_coarse; j++){
    for (long n=0; n<N; n++){
      truth[n+j*N] = cos((double)(n+1)*j*dt_coarse) / ((double)n+1);
    }
  }

  // printf("IC is [%f,%f]\n",u_fine[0],u_fine[1]);
  // printf("init truth is [%f,%f]\n",truth[0],truth[1]);

  printf("\n");
  /******************************************************************
  **************************** Coarse BE ****************************
  ******************************************************************/
  printf("********** Coarse Solver (dt=%2.1E) **********\n",dt_coarse);
  //time coarse solve
  double timer_coarse = omp_get_wtime();
  for (long s=0; s< N_coarse; s++){
    backEuler(myRHS,timeVec[s+1],timeVec[s],dt_coarse,u_coarse+((s+1)*N),
              coarse_righthand_buff,coarse_FPsoln_buff,coarse_FPrighthand_buff,N);
  }
  timer_coarse = omp_get_wtime() - timer_coarse;

  //compute coarse solve error;
  double coarse_err = 0;
  for(long s=0; s < N*(N_coarse+1); s++){
    coarse_err = fmax(std::abs(truth[s] - u_coarse[s]), coarse_err);
  }
  //printf("coarse solve using dt=%2.2e took: %f s\n",dt_coarse,timer_coarse);
  //printf("max coarse solver error is: %6.4e \n", coarse_err);
  //printf("coarse solver (dt=%2.1E) time: %f s\n",dt_coarse,timer_coarse);
  printf("coarse solver time: %f s\n",timer_coarse);
  printf("coarse solver error: %6.4E \n", coarse_err);
  printf("\n");

  /****************************************************************
  **************************** Fine BE ****************************
  ****************************************************************/
  printf("********** Fine Solver (dt=%2.1E) **********\n",dt_fine);
  //time fine solve
  double timer_fine = omp_get_wtime();
  for (long s=0; s< N_coarse; s++){
    backEuler(myRHS,timeVec[s+1],timeVec[s],dt_fine,u_fine+((s+1)*N),
              fine_righthand_buff,fine_FPsoln_buff,fine_FPrighthand_buff,N);
  }
  timer_fine = omp_get_wtime() - timer_fine;

  //compute fine solve error
  double fine_err = 0;
  for(long s=0; s < N*(N_coarse+1); s++){
    fine_err = fmax(std::abs(truth[s] - u_fine[s]), fine_err);
  }

  //printf("fine solve using dt=%2.2e took: %f s\n",dt_fine,timer_fine);
  //printf("max fine solver error is: %6.4e \n", fine_err);
  //printf("fine solver (dt=%2.1E) time: %f s\n",dt_fine,timer_fine);
  printf("fine solver time: %f s\n",timer_fine);
  printf("fine solver error: %6.4E \n", fine_err);
  printf("\n");

  // printf("truth end: [%f,%f]\n",truth[N*(N_coarse+1)-2],truth[N*(N_coarse+1)-1]);
  // printf("u_fine end: [%f,%f]\n",u_fine[N*(N_coarse+1)-2],u_fine[N*(N_coarse+1)-1]);


  //solve and time
  // double t_timer = omp_get_wtime();
  // u_fine = backEuler(myRHS,T,0,dt/M,1);
  // t_timer = omp_get_wtime() - t_timer;
  // printf("error at terminal time T=%6.2f is %6.3e\n",T,std::abs(truth-u));
  // printf("time elapse: %f s\n",t_timer);
  // printf("Final solution is u=%f and T=%f\n",u,T);

  free(timeVec);
  free(u_coarse);
  free(u_fine);
  free(truth);

  free(coarse_righthand_buff);
  free(coarse_FPsoln_buff);
  free(coarse_FPrighthand_buff);
  free(fine_righthand_buff);
  free(fine_FPsoln_buff);
  free(fine_FPrighthand_buff);

  return 0;
}
