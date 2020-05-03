// kdv with parareal, SBDF2 is high-fidelity, IMEX Euler is low-fidelity
// g++-9 -std=c++11 -fopenmp -O2 -o parareal-kdv parareal-kdv.cpp && ./parareal-kdv
#include <stdio.h>
#include <math.h>
#include <cstdlib>
#include <omp.h>
#include <complex>

using namespace std ;
typedef complex<double> dcomp;

#define N_THREADS_PR 64
#define N_THREADS_FFT 1

long depth;
double pi = 4.0*atan(1);
dcomp i = sqrt((dcomp) -1); //unit imaginary
long N;
long mem_size;

dcomp complex_exp_taylor(const double x){

  double C1 = 1.0;
  double C2 = x;
  double val1 = 1.0;
  double val2 = x;
  long s = 1.0;
  double xsq = -0.5*x*x;

  while(fabs(C1) > 1e-14){
    C1 = C1*xsq/(2.0*s*s - s);
    C2 = C2*xsq/(2.0*s*s + s);
    val1 += C1;
    val2 += C2;
    s++;
  }

  return val1 + i*val2;

}

/* Computes the DFT of x and writes to x_hat
:q * x: input array of grid values
 * x_hat: output array of Fourier coefficients
 * N: size of current array (changes in recursion)
 * Nmodes: size of full array
*/
void fft(const dcomp* x,dcomp* x_hat, const long N, const long Nmodes){

  if(N == 1) x_hat[0] = x[0];
  else{

    long Nmod2 = N/2;

    // Preallocate local x_hat array
    dcomp* x_hat_loc = (dcomp*) malloc(N * sizeof(dcomp));

    for(long s = 0; s < Nmod2; s++){
      x_hat_loc[s] = x[2*s]; //even grid points
      x_hat_loc[s+Nmod2] = x[2*s+1]; //odd grid points
    }

    // Recursive step
    #pragma omp task if(N > Nmodes/(depth+1))
    fft(x_hat_loc,x_hat_loc,Nmod2,Nmodes);
    #pragma omp task if(N > Nmodes/(depth+1))
    fft(x_hat_loc + Nmod2,x_hat_loc + Nmod2,Nmod2,Nmodes);
    #pragma omp taskwait

    dcomp xnhat, t;
    dcomp w = complex_exp_taylor(-pi/Nmod2);
    dcomp C = 1.0;

    // Reassebmle using the Cooley-Tukey algorithm
    for(long s = 0; s < Nmod2; s++){
      t = x_hat_loc[s];
      xnhat = C*x_hat_loc[s+Nmod2];
      x_hat_loc[s] = t + xnhat;
      x_hat_loc[s+Nmod2] = t - xnhat;
      C = C*w;
    }

    for(long s = 0; s < N; s++) x_hat[s] = x_hat_loc[s];
    free(x_hat_loc);

  }
}

/* Computes the inverse DFT of x_hat
 * x: ouput array of grid values
 * x_hat: input array of Fourier coefficients
 * N: size of current array (changes in recursion)
 * Nmodes: size of full array
*/
void ifft(dcomp* x,const dcomp* x_hat,const long N,const long Nmodes){

  if(N == 1) x[0] = x_hat[0]/((dcomp) Nmodes);
  else{

    long Nmod2 = N/2;

    // Preallocate local x array
    dcomp* x_loc = (dcomp*) malloc(N * sizeof(dcomp));

    for(long s = 0; s < Nmod2; s++){
      x_loc[s] = x_hat[2*s];
      x_loc[s + Nmod2] = x_hat[2*s+1];
    }

    // Recursive step
    #pragma omp task if(N > Nmodes/(depth+1))
    ifft(x_loc,x_loc,Nmod2,Nmodes);
    #pragma omp task if(N > Nmodes/(depth+1))
    ifft(x_loc+Nmod2,x_loc+Nmod2,Nmod2,Nmodes);
    #pragma omp taskwait

    dcomp xn, t;
    dcomp w = complex_exp_taylor(pi/Nmod2);
    dcomp C = 1.0;

    // Reassemble using the Cooley-Tukey algorithm
    for(long s = 0; s < Nmod2; s++){
      t = x_loc[s];
      xn = C*x_loc[s+Nmod2];
      x_loc[s] = t + xn;
      x_loc[s+Nmod2] = t - xn;
      C = C*w;
    }

    // Write sums to full array
    for(long s = 0; s < N; s++) x[s] = x_loc[s];
    free(x_loc);

  }
}

/* Nonlinear part of KdV in Fourier space
 * u_hat: input array of Fourier coefficients
 * B_hat: output array of d/dx(3.0 * u^2) in Fourier space
 * ikx: array of Fourier modes
 * N: number of modes
 */
void kdv_rhs_hat(dcomp* u_hat, dcomp* B_hat, dcomp* ikx, long N){

  // Allocate memory
  //dcomp* usq = (dcomp*) malloc(N * sizeof(dcomp));
  double usq_s;

  // Compute u in real space (we overwrite B_hat)
  #pragma omp parallel num_threads(N_THREADS_FFT)
  {
    #pragma omp single
    ifft(B_hat,u_hat,N,N);

    // Square value
    #pragma omp for private(usq_s)
    for(long s = 0; s < N; s++){
      usq_s = real(B_hat[s]);
      B_hat[s] = 3*usq_s*usq_s;
    }

    // Compute the Fourier transform of 3*u^2
    #pragma omp single
    fft(B_hat,B_hat,N,N);

    // Differentiate in Fourier space
    #pragma omp for
    for(long s = 0; s < N; s++)
      B_hat[s] *= ikx[s];
  }

  //free(usq);

}


////////////////////////////////////////////////////////////////////////

/*
SBDF2_kdv
Solve dy/dt = A*y + B(y)
Specialized for the KdV equation in Fourier space
A is a diagonal matrix = diag(-(ikx)^3), where ikx are i times the Fourier modes
B is the nonlinearity, B(y) = 3*ikx .* fft( (ifft(y)).^2 )
Tntegrate from t0 to tfinal with step size dt.
Initial value is y0 at time t0.
Write solution into y (y at time tfinal).
Memory addresses B_hat and B_hatm1 are N-vectors to be used for the nonlinear terms.
*/
void SBDF2_kdv(dcomp* y, double t0, double tfinal,
               double dt, dcomp* y0, dcomp* ikx, long N,
               dcomp* B_hat, dcomp* B_hatm1){

    // initialize method parameters
    long M = (long) floor((tfinal - t0) / dt + 0.1); // number of time steps to integrate

    // initial solution: use the same memory as y0
    // change in name just for readability
    dcomp* ym1 = y0;

    // initial nonlinearity; store in B_hatm1
    kdv_rhs_hat(y0, B_hatm1, ikx, N);

    /*
    Take an Euler predictor/corrector step (Heun's method) -- write to y.
    We use a second order method to maintain asymptotic convergence.
    */
    // predictor step
    dcomp ikx_j;
    #pragma omp parallel for schedule(static) num_threads(N_THREADS_FFT) private(ikx_j)
    for(long j = 0; j < N; j++){
      ikx_j = ikx[j];
      y[j] = 1.0/(1.0 + dt*ikx_j*ikx_j*ikx_j)*(y0[j] + dt*B_hatm1[j]);
    }

    // corrector step
    kdv_rhs_hat(y, B_hat, ikx, N);
    #pragma omp parallel for schedule(static) num_threads(N_THREADS_FFT) private(ikx_j)
    for(long j = 0; j < N; j++){
      ikx_j = ikx[j];
      y[j] = 1.0/(1.0 + 0.5*dt*ikx_j*ikx_j*ikx_j)*(y0[j] + 0.5*dt*(B_hatm1[j] - ikx_j*ikx_j*ikx_j*y[j] + B_hat[j]));
    }

    // IMEX_Euler_kdv(y,t0,t0+dt,dt,ym1,ikx,N,B_hat);

    // main loop; time steps
    for(long m = 0; m < M-1; m++) {
      
      printf("Iteration %d/%d\r", m+1,M);

      // compute the nonlinear component; store in B_hat
      // done using OpenMP parallel (I)FFT
      kdv_rhs_hat(y, B_hat, ikx, N);

      // take one update step; update each of the N components
      #pragma omp parallel for schedule(static) num_threads(N_THREADS_FFT) private(ikx_j)
      for(long j = 0; j < N; j++) {

        // make a copy of the current iterate
        // needed for parsimonious memory management at the update step
        dcomp yj_temp = y[j];
        dcomp ym1_j = yj_temp;
        dcomp B_hat_j = B_hat[j];
        ikx_j = ikx[j];

        // using the TRBDF2 method iteration
        // debugging printouts
        // printf("y[%d] = %f\n", j, y[j]);
        // printf("ym1[j] = %f\n", ym1[j]);
        // printf("B_hat[j] = %f\n", B_hat[j]);
        // printf("B_hatm1[j] = %f\n", B_hatm1[j]);
        yj_temp = 4.0*yj_temp - ym1[j] + 2.0*dt*(2.0*B_hat_j - B_hatm1[j]);
        y[j] = yj_temp / ( 3.0 + 2.0*dt*ikx_j*ikx_j*ikx_j); // simplified a double negative

        // update previous solution and nonlinearity
        ym1[j] = ym1_j;
        B_hatm1[j] = B_hat_j;

      } // component for

    } // time step for
}


void IMEX_Euler_kdv(dcomp* y, // solution to write to
                    double t0, // initial time
                    double tfinal, //final time
                    double dt, //time step
                    dcomp* y0,
                    dcomp* ikx, //constant array of ikx
                    long N, // dimension of the ODE
                    dcomp* B_hat //buffer for non-stiff portion of KdV
                    ){

  long M = floor((tfinal - t0) / dt + 0.1);
  dcomp ikx_j;

  for (long j = 0; j < N; j++) y[j] = y0[j];

  //assume y has come preloaded with the initial condition.
  for (long s = 0; s < M; s++){
    kdv_rhs_hat(y,B_hat,ikx,N); //compute the RHS B

    //#pragma omp parallel for num_threads(N_THREADS_FFT) firstprivate(ik_temp);
    for (long j = 0; j < N; j++){
      ikx_j = ikx[j]; //get current coordinate ik (firstprivate)
      y[j] = (y[j] + dt*B_hat[j])/(1.0 + dt*ikx_j*ikx_j*ikx_j); // write back to y
    }
  }
}

void parareal(dcomp* u, //solution to write to, size (M+1) x N
              dcomp* u_par, // copy of u for threads to write into, size (M+1) x N
              double* timeVec, //vector of times, size M+1
              dcomp* ikx, //ikx vector
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
  #pragma omp parallel num_threads(N_THREADS_PR)
  {//begin parallel region

    long tid = omp_get_thread_num(); // get the thread
    // initialize thread-private buffers
    dcomp* writeto = (dcomp*) malloc(mem_size); //to write to
    dcomp* B_buff = (dcomp*) malloc(mem_size); // to hold B for TRBDF2
    dcomp* Bm1_buff = (dcomp*) malloc(mem_size); // to hold Bm1 for TRBDF2

    //initialize
    while (counter<max_iter){

      //compute multiple shooting in parallel
      // Optimized memory access using pointer arithmetic.
      // By calling Euler first, you avoid overwriting u so
      // you can reuse it as the initial condition in SBDF2
      #pragma omp for
      for(long s = 0; s < M; s++){
        IMEX_Euler_kdv(writeto,timeVec[s],timeVec[s+1],dtG,u + s*N,ikx,N,B_buff); // solve and write
        SBDF2_kdv(u_par + (s+1)*N,timeVec[s],timeVec[s+1],dtF,u + s*N,ikx,N,B_buff,Bm1_buff); // solve and write
        for (long j=0; j<N; j++) u_par[j+(s+1)*N] -= writeto[j]; //update u_par
     
      }

      //compute update + extra Gpropagator solve in serial
      if (0==tid){ //master thread does serial
        for (long s=0; s < M ;s++){ //loop over every time step
          IMEX_Euler_kdv(writeto,timeVec[s],timeVec[s+1],dtG,u_par + s*N,ikx,N,B_buff); // solve and write
          for (long j=0; j<N; j++)
            u_par[j+(s+1)*N] += writeto[j]; //parareal correction
        }
        counter++; //only tid0 updates counter
      }

      #pragma omp barrier
      #pragma omp for
      for(long j = 0; j < N*(M+1); j++) u[j] = u_par[j]; //write from u_par to u

    } // end of while loop

    free(writeto);
    free(B_buff);
    free(Bm1_buff);
  } //end of parallel region

}


int main(int argc, char** argv){

  double t0 = 0; // initial time
  double tfinal = 0.64; // terminal time
  double dt;
  double ratio;
  long PR_iters;

  if(argc < 5){
    printf("\n Call instructions:\n    ./SBDF2-test (N) (dt) (ratio) (PR iters) \n\n");
    return 0;
  }
  else{
    N = atoi(argv[1]);
    dt = atof(argv[2]);
    ratio = atof(argv[3]);
    PR_iters = atof(argv[4]);
  }

  long M = floor((tfinal - t0) / dt + 0.1);
  //get vector of times
  double *timeVec = (double*) malloc((M+1) * sizeof(double));
  for(long s=0; s<= M; s++) timeVec[s] = (double)s*dt;

  printf("t0 = %f\ntfinal = %f\ncoarse dt = %f\nfine dt = %f\n\n", t0,tfinal,dt,dt/(double)ratio);

  depth = (long) log2(N_THREADS_FFT);
  double L = 60; //size of domain
  double h = L/N; // step size

  printf("Using %d threads for FFT\n",(long) pow(2.0,depth));
  printf("Using %d threads for parareal\n", N_THREADS_PR);
  printf("Using %d threads in total\n",N_THREADS_PR*N_THREADS_FFT);
  mem_size = (N+1) * sizeof(dcomp); //add a buffer at the end

  // Preallocate arrays
  dcomp* u_coarse      = (dcomp*) malloc((M+1)*mem_size); // numerical ODE solution
  dcomp* u_hat_coarse  = (dcomp*) malloc((M+1)*mem_size); //Fourier coefficients
  dcomp* u_fine        = (dcomp*) malloc((M+1)*mem_size); // numerical ODE solution
  dcomp* u_hat_fine    = (dcomp*) malloc((M+1)*mem_size); //Fourier coefficients
  dcomp* u_PR          = (dcomp*) malloc((M+1)*mem_size); // numerical ODE solution
  dcomp* u_hat_PR      = (dcomp*) malloc((M+1)*mem_size); //Fourier coefficients
  dcomp* u_hat_PR_buff = (dcomp*) malloc((M+1)*mem_size); //Fourier coefficients
  dcomp* u_hatm1       = (dcomp*) malloc(mem_size); // solution at previous time step
  dcomp* u_hat0        = (dcomp*) malloc(mem_size); //Fourier coefficients
  dcomp* u_true        = (dcomp*) malloc((M+1)*mem_size); //reference solution
  dcomp* u_hat_true    = (dcomp*) malloc((M+1)*mem_size); //reference in Fourier space solution
  dcomp* B_hat         = (dcomp*) malloc(mem_size); //nonlinear part of KdV
  dcomp* B_hatm1       = (dcomp*) malloc(mem_size); //Fourier coefficients
  dcomp* ikx           = (dcomp*) malloc(mem_size); //Fourier modes
  double* xx           = (double*) malloc(N * sizeof(double)); //grid

  /*
  Test problem -- a soliton wave
  */
  // test function u(x,t) = -(c/2) * sech^2( sqrt(c)/2 * (x-c*t) )
  dcomp c = 1.0;
  dcomp arg;
  for(long s = 0; s < N; s++){

    xx[s] = (L/N)*s - L/2; //grid point values on [0,L]

    // initial data
    arg = sqrt(c)/2.0 * (xx[s] - c*t0);
    u_coarse[s] = -(c/2.0) * pow(cosh(arg), -2.0); //assign function values (memory address u is a placeholder)
    u_fine[s] = u_coarse[s];

    if(s < N/2) ikx[s] = 2*pi*i/L*((dcomp) s); //modes 0,...,N/2-1
    else ikx[s] = 2*pi*i/L*((dcomp) s - (dcomp) N); //modes -N/2,..,-1
  }

  //build true solution at all times
  for (long j=0; j<=M; j++){
    for (long s=0; s<N; s++){
        // the true solution at all times
        arg = sqrt(c)/2.0 * (xx[s] - c*timeVec[j]);
        u_true[s+j*N] = -(c/2.0) * pow(cosh(arg), -2.0);
    }
  }

  //get true solution in Fourier space
  for (long j=0; j<=M; j++)
    fft(u_true+(j*N), u_hat_true+(j*N), N, N); // True solution in Fourier space

  //get the IC in fourier space
  fft(u_coarse, u_hat_coarse, N, N); // solve the ODE in Fourier space. Put initial data there.
  fft(u_fine, u_hat_fine,N,N);

  printf("\n");
  printf("******************** RUNNING SOLVERS ********************\n");
  // coarse solve in Fourier space
  double tt_coarse = omp_get_wtime();
  for(long s=0; s<M; s++){
    for(long j=0; j<N; j++) 
      u_hat0[j] = u_hat_coarse[j+s*N]; //copy correct IC to a buffer u_hat0
    IMEX_Euler_kdv(u_hat_coarse+((s+1)*N), timeVec[s], timeVec[s+1], dt, u_hat0, ikx, N, B_hat); //solve)
  }
  tt_coarse = omp_get_wtime() - tt_coarse;
  printf("Coarse time = %1.4e\n",tt_coarse);

  // fine solve in Fourier space
  double tt_fine = omp_get_wtime();
  for(long s=0; s<M; s++){
    for(long j=0; j<N; j++)
      u_hat0[j] = u_hat_fine[j+s*N]; //copy correct IC to a buffer u_hat0
    SBDF2_kdv(u_hat_fine+((s+1)*N), timeVec[s], timeVec[s+1], dt/ratio, u_hat0, ikx, N, B_hat, B_hatm1); //solve
  }
  tt_fine = omp_get_wtime() - tt_fine;
  printf("Fine time = %1.4e\n",tt_fine);

  //parareal solve in Fourier space
  for (long s=0; s<(M+1)*N; s++){
    u_hat_PR[s] = u_hat_coarse[s];
    u_hat_PR_buff[s] = u_hat_coarse[s];
  }
  
  double tt_PR = omp_get_wtime();
  parareal(u_hat_PR,u_hat_PR_buff,timeVec,ikx,M,dt/ratio,dt,PR_iters,N);
  tt_PR = omp_get_wtime() - tt_PR;
  printf("PR time = %1.4e\n",tt_PR);

  // recover the solutions in real space via ifft
  for (long s=0; s<=M; s++){
    ifft(u_coarse+(s*N), u_hat_coarse+(s*N), N, N);
    ifft(u_fine+(s*N), u_hat_fine+(s*N), N, N);
    ifft(u_PR+(s*N), u_hat_PR+(s*N), N, N);
  }

  // debugging printouts
  // for(long s = 0; s < N; s++) printf("u_hat[%d] = %f\n", s, u_hat[s]);
  // for(long s = 0; s < N; s++) printf("u_hat_true[%d] = %f\n", s, u_hat_true[s]);

  // compute errors
  double err_fine = 0.0;
  double err_coarse = 0.0;
  double err_PR = 0.0;
  double temp_err_coarse;
  double temp_err_fine;
  double temp_err_PR;
 
  for(long s=0; s<=M; s++){
    temp_err_coarse = 0;
    temp_err_fine = 0;
    temp_err_PR = 0;
    for (long j=0; j<N; j++){
      temp_err_coarse += h*abs(u_coarse[j+s*N] - u_true[j+s*N]);
      temp_err_fine += h*abs(u_fine[j+s*N] - u_true[j+s*N]);
      temp_err_PR += h*abs(u_PR[j+s*N] - u_true[j+s*N]);
    }
    if( s % ((long) (0.01/dt)) == 0 )
      //printf("PR error at time %2.6f is: %1.4e\n",timeVec[s],temp_err_PR);
    err_fine = fmax(temp_err_fine,err_fine);
    err_fine = temp_err_fine;
    err_coarse = fmax(temp_err_coarse,err_coarse);
    err_coarse = temp_err_coarse;
    err_PR = fmax(temp_err_PR,err_PR);
    err_PR = temp_err_PR;
  }

  double err_fine2 = 0.0;
  double err_coarse2 = 0.0;
  double err_PR2 = 0.0;
  for(long s=0; s<=M; s++){
    temp_err_coarse = 0;
    temp_err_fine = 0;
    temp_err_PR = 0;
    for (long j=0; j<N; j++){
      temp_err_coarse += h*abs(u_hat_coarse[j+s*N] - u_hat_true[j+s*N]);
      temp_err_fine += h*abs(u_hat_fine[j+s*N] - u_hat_true[j+s*N]);
      temp_err_PR += h*abs(u_hat_PR[j+s*N] - u_hat_true[j+s*N]);
    }
    err_fine2 = fmax(temp_err_fine,err_fine2);
    err_coarse2 = fmax(temp_err_coarse,err_coarse2);
    err_PR2 = fmax(temp_err_PR,err_PR2);
  }

   printf("\n");
   printf("******************** ERRORS ********************\n");

   printf("Coarse Error: %1.10e\n", err_coarse);
   printf("Coarse Fourier Error: %1.10e\n", err_coarse2);
   printf("\n");
   printf("Fine Error: %1.10e\n",err_fine);
   printf("Fine Fourier Error: %1.10e\n",err_fine2);
   printf("\n");
   printf("Parareal Error: %1.10e\n",err_PR);
   printf("Parareal Fourier Error: %1.10e\n",err_PR2);
   printf("\n");

   //measure speedup and efficiency
   double speedup = tt_fine / tt_PR;
   double PR_speedup = N_THREADS_PR/(double)PR_iters;
   double ideal_speedup = (N_THREADS_PR*N_THREADS_FFT)/(double)PR_iters;
   double efficiency = speedup / ideal_speedup;
   printf("speedup: %f\n",speedup);
   printf("PR efficiency: %f\n",speedup/PR_speedup);
   printf("total efficiency: %f\n",efficiency);
   printf("\n");

  // tidy
  free(u_coarse);
  free(u_hat_coarse);
  free(u_fine);
  free(u_hat_fine);
  free(u_PR);
  free(u_hat_PR);
  free(u_hat_PR_buff);
  free(u_hatm1);
  free(u_hat0);
  free(u_true);
  free(u_hat_true);
  free(B_hat);
  free(B_hatm1);
  free(ikx);
  free(xx);
  free(timeVec);

  return 0;
}
