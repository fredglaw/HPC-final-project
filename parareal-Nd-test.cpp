// test script to test parareal with backward Euler
// g++-9 -std=c++11 -fopenmp -O2 -o parareal-Nd-test parareal-Nd-test.cpp && ./parareal-Nd-test
#include <stdio.h>
#include <cmath>
#include <cstdlib>
#include <omp.h>

#define N_THREADS 2


void sinRHS (double* right, double t, double* x, long N){
  //RHS which just gives sin
  // for (long k=0; k<N; k++) right[k] = -1*sin((k+1.)*t);
  for (long k=0; k<N; k++) right[k] = -1*sin((k+1.)*t) + sqrt(k+1.);
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





void parareal(void (*RH_handle) (double*,double,double*,long),
              double* u, //solution to write to, size (M+1) x N
              double* u_par, // copy of u for threads to write into, size (M+1) x N
              double* timeVec, //vector of times, size M+1
              long M, //number of time steps to evaluate solution
              double dtF, //time step for F propagator
              double dtG, //time step for G propagator
              long max_iter, //number of iterations to do
              long N //dimension of the ODE
              ){
  /*
  Parareal function using backward Euler with two time steps
  */
  long counter=0; //iteration counter
  //run parareal iterations
  #pragma omp parallel num_threads(N_THREADS)
  {//begin parallel region
    int tid = omp_get_thread_num(); // get the thread
    double* thread_parvec = (double*) malloc(N*sizeof(double));
    double* thread_righthand = (double*) malloc(N*sizeof(double));
    double* thread_uFP = (double*) malloc(N*sizeof(double));
    double* thread_f_uFP = (double*) malloc(N*sizeof(double));

    //initialize
    for (long j=0; j<N; j++){
      thread_parvec[j] = 0;
      thread_righthand[j] = 0;
      thread_uFP[j] = 0;
      thread_f_uFP[j] = 0;
    }

    while (counter<max_iter){
      //compute multiple shooting in parallel
      #pragma omp for
      for(long s=0; s < M; s++){
        //F-propagator step
        for (long j=0; j<N; j++) thread_uFP[j] = u[j+s*N]; //copy the current solution to FP buffer
        backEuler(RH_handle,timeVec[s+1],timeVec[s],dtF,thread_parvec,thread_righthand,
                  thread_uFP,thread_f_uFP,N); //expensive BE solve, write to thread_parvec
        for (long j=0; j<N; j++) u_par[j+(s+1)*N] = thread_parvec[j]; //update u_par

        //G-propagator step
        for (long j=0; j<N; j++) thread_uFP[j] = u[j+s*N]; //copy the current solution to FP buffer, again
        backEuler(RH_handle,timeVec[s+1],timeVec[s],dtG,thread_parvec,thread_righthand,
                  thread_uFP,thread_f_uFP,N); //cheap BE solve, write to thread_parvec
        for (long j=0; j<N; j++) u_par[j+(s+1)*N] -= thread_parvec[j]; //update u_par
      }

      #pragma omp barrier

      //compute update + extra Gpropagator solve in serial
      if (0==tid){ //master thread does serial
        for (long s=0; s<M ;s++){ //loop over every time step
          for (long j=0; j<N; j++) thread_uFP[j] = u_par[j+s*N]; //copy to tid0 buffer
          backEuler(RH_handle,timeVec[s+1],timeVec[s],dtG,thread_parvec,thread_righthand,
                    thread_uFP,thread_f_uFP,N); // sequential cheap BE, write to thread_parvec
          for (long j=0; j<N; j++) u_par[j+(s+1)*N] += thread_parvec[j]; //parareal correction
        }
        counter++; //only tid0 updates counter
      }

      #pragma omp barrier

      #pragma omp for
      for(long s=N; s<N*(M+1); s++) u[s] = u_par[s]; //write from u_par to u

    } // end of while loop


    free(thread_parvec);
    free(thread_righthand);
    free(thread_uFP);
    free(thread_f_uFP);
  } //end of parallel region

}







int main(){
  double T = 10; // terminal time
  double ratio = 1e5; //ratio of course to fine
  double dt_coarse = 1; //coarse time step
  double dt_fine = dt_coarse/ratio; // fine time step
  long N = 100;
  long K = 1; //number of parareal iterations

  long N_coarse = round(T/dt_coarse); //number of coarse time steps

  //timeVec is the vector of times to get solution at
  double *timeVec = (double*) malloc((N_coarse+1) * sizeof(double));
  for(long s=0; s<= N_coarse; s++) timeVec[s] = (double)s*dt_coarse;

  // where I save the solution using the coarse solver and the fine solver
  double* u_coarse = (double*) malloc(N*(N_coarse+1) * sizeof(double));
  double* u_fine = (double*) malloc(N*(N_coarse+1) * sizeof(double));
  double* u_PR = (double*) malloc(N*(N_coarse+1) * sizeof(double));
  double* u_PR_copy = (double*) malloc(N*(N_coarse+1) * sizeof(double));
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
  for (long n=0; n<N; n++){ //initialize all buffers
    u_coarse[n] = 1./((double)n+1);
    u_fine[n] = 1./((double)n+1);
    coarse_righthand_buff[n] = 0;
    fine_righthand_buff[n] = 0;
    coarse_FPsoln_buff[n] = u_coarse[n];
    fine_FPsoln_buff[n] = u_fine[n];
    coarse_FPrighthand_buff[n] = 0;
    fine_FPrighthand_buff[n] = 0;
  }

  // solve for a -sin RHS and set initial conditions
  void (*myRHS) (double*,double,double*,long) = sinRHS;

  //get truth
  for (long j=0; j<= N_coarse; j++){
    for (long n=0; n<N; n++){
      // truth[n+j*N] = (cos((double)(n+1)*j*dt_coarse) / ((double)n+1));
      truth[n+j*N] = (cos((double)(n+1)*j*dt_coarse) / ((double)n+1)) + sqrt(n+1.)*j*dt_coarse;
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

  /*****************************************************************
  **************************** Parareal ****************************
  *****************************************************************/
  printf("********** Parareal (%d threads) **********\n",N_THREADS);
  //set up parareal
  for (long s=0; s< N*(N_coarse+1); s++){
    u_PR[s] = u_coarse[s]; //copy coarse
    u_PR_copy[s] = u_coarse[s];
  }

  //time parareal solve
  double timer_parareal = omp_get_wtime();
  parareal(myRHS,u_PR,u_PR_copy,timeVec,N_coarse,dt_fine,dt_coarse,K,N);
  timer_parareal = omp_get_wtime() - timer_parareal;


  //compute parareal error
  double parareal_err = 0;
  for(long s=0; s < N*(N_coarse+1); s++){
    parareal_err = fmax(std::abs(truth[s] - u_PR[s]), parareal_err);
    //printf("error is: %f\n",std::abs(truth[s] - u_PR[s]));
    //printf("PR solution is: %f\n",u_PR[s]);
  }

  printf("PR %d iterations time: %f s\n",K,timer_parareal);
  printf("PR error: %6.4E \n", parareal_err);

  //compute difference of parareal from fine solve
  double parareal_fine_diff = 0;
  for(long s=0; s < N*(N_coarse+1); s++){
    parareal_fine_diff = fmax(std::abs(u_fine[s] - u_PR[s]), parareal_fine_diff);
  }
  // printf("max parareal and fine solver difference is: %6.4e \n", parareal_fine_diff);
  printf("max PR and fine solver difference: %6.4E \n", parareal_fine_diff);

  //measure speedup and efficiency
  double speedup = timer_fine / timer_parareal;
  double ideal_speedup = N_THREADS/(double)K;
  double efficiency = speedup / ideal_speedup;

  //printf("speedup is %f, efficiency is %f\n",speedup,efficiency);
  printf("speedup: %f\n",speedup);
  printf("efficiency: %f\n",efficiency);
  printf("\n");

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
  free(u_PR);
  free(u_PR_copy);
  free(truth);

  return 0;
}
