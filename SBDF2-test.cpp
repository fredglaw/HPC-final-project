// g++ -std=c++11 -O3 -march=native -fopenmp TRBDF2-test.cpp -o TRBDF2-test

#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <complex>

using namespace std ;
typedef complex<double> dcomp;

//int depth = 2.0; // 2^depth threads will be used
int depth;
double pi = 4.0*atan(1);
dcomp i = sqrt((dcomp) -1); //unit imaginary


////////////////////////////////////////////////////////////////////////
/*
Everything below is fft.cpp, followed by TRBDF2_kdv() and main()
*/


/* Computes the DFT of x and writes to x_hat
 * x: input array of grid values
 * x_hat: output array of Fourier coefficients
 * N: size of current array (changes in recursion)
 * Nmodes: size of full array
*/
void fft(dcomp* x,dcomp* x_hat, int N, int Nmodes){

  if(N == 1) x_hat[0] = x[0];
  else{

    // Preallocate some arrays
    dcomp* x1 = (dcomp*) malloc(N/2 * sizeof(dcomp));
    dcomp* x2 = (dcomp*) malloc(N/2 * sizeof(dcomp));
    dcomp* x1_hat = (dcomp*) malloc(N/2 * sizeof(dcomp));
    dcomp* x2_hat = (dcomp*) malloc(N/2 * sizeof(dcomp));

    for(int s = 0; s < N/2; s++){
      x1[s] = x[2*s]; //even grid points
      x2[s] = x[2*s+1]; //odd grid points
    }

    // Recursive step
    #pragma omp task if(N > Nmodes/(depth+1))
    fft(x1,x1_hat,N/2,Nmodes);
    #pragma omp task if(N > Nmodes/(depth+1))
    fft(x2,x2_hat,N/2,Nmodes);
    #pragma omp taskwait

    // Write sums to full array
    for(int s = 0; s < N/2; s++){
      x_hat[s] = x1_hat[s];
      x_hat[s+N/2] = x2_hat[s];
    }

    dcomp arg, xnhat, t;

    // Reassebmle using the Cooley-Tukey algorithm
    for(int s = 0; s < N/2; s++){
      t = x_hat[s];
      arg = -2*pi*s/N;
      xnhat = exp(i*arg)*x_hat[s+N/2];
      x_hat[s] = t + xnhat;
      x_hat[s+N/2] = t - xnhat;
    }

    // Clean up
    free(x1); free(x2);
    free(x1_hat); free(x2_hat);

  }
}

/* Computes the inverse DFT of x_hat
 * x: ouput array of grid values
 * x_hat: input array of Fourier coefficients
 * N: size of current array (changes in recursion)
 * Nmodes: size of full array
*/
void ifft(dcomp* x,dcomp* x_hat, int N, int Nmodes){

  if(N == 1) x[0] = x_hat[0]/((dcomp) Nmodes);
  else{

    // Preallocate some arrays
    dcomp* x1 = (dcomp*) malloc(N/2 * sizeof(dcomp));
    dcomp* x2 = (dcomp*) malloc(N/2 * sizeof(dcomp));
    dcomp* x1_hat = (dcomp*) malloc(N/2 * sizeof(dcomp));
    dcomp* x2_hat = (dcomp*) malloc(N/2 * sizeof(dcomp));

    for(int s = 0; s < N/2; s++){
      x1_hat[s] = x_hat[2*s]; //get even modes
      x2_hat[s] = x_hat[2*s+1]; //get odd modes
    }

    // Recursive step
    #pragma omp task if(N > Nmodes/(depth+1))
    ifft(x1,x1_hat,N/2,Nmodes);
    #pragma omp task if(N > Nmodes/(depth+1))
    ifft(x2,x2_hat,N/2,Nmodes);
    #pragma omp taskwait

    // Write sums to full array
    for(int s = 0; s < N/2; s++){
      x[s] = x1[s];
      x[s+N/2] = x2[s];
    }

    dcomp arg, xn, t;

    // Reassemble using the Cooley-Tukey algorithm
    for(int s = 0; s < N/2; s++){
      t = x[s];
      arg = 2*pi*s/N;
      xn = exp(i*arg)*x[s+N/2];
      x[s] = t + xn;
      x[s+N/2] = t - xn;
    }

    // Clean up
    free(x1); free(x2);
    free(x1_hat); free(x2_hat);

  }
}

// Differentiation in Fourier space
void sp_dx(dcomp* u, dcomp* ux, dcomp* ikx, int N){

  dcomp* u_hat = (dcomp*) malloc(N * sizeof(dcomp));
  #pragma omp parallel
  {
    #pragma omp single
    fft(u,u_hat,N,N);
  }

  for(int s = 0; s < N; s++)
    u_hat[s] = ikx[s]*u_hat[s];

  ifft(ux,u_hat,N,N);
  free(u_hat);

}

/* Nonlinear part of KdV equation
 * u: input array of grid values
 * B: output array of d/dx ( 3.0 * u^2 )
 * ikx: array of Fourier modes
 * N: number of modes
 */
void kdv_rhs(dcomp* u, dcomp* B, dcomp* ikx, int N){

  // Allocate memory
  dcomp* B_hat = (dcomp*) malloc(N * sizeof(dcomp)); //nonlinear term

  // Square values
  for(int s = 0; s < N; s++)
    B[s] = 3.0*u[s]*u[s];

  #pragma omp parallel
  {
    #pragma omp single
    fft(B,B_hat,N,N);
  }

  // Differentiate in Fourier space
  for(int s = 0; s < N; s++)
    B_hat[s] = ikx[s]*B_hat[s];

  ifft(B,B_hat,N,N);
  free(B_hat);

}

/* Nonlinear part of KdV in Fourier space
 * u_hat: input array of Fourier coefficients
 * B_hat: output array of d/dx(3.0 * u^2) in Fourier space
 * ikx: array of Fourier modes
 * N: number of modes
 */
void kdv_rhs_hat(dcomp* u_hat, dcomp* B_hat, dcomp* ikx, int N){

  // Allocate memory
  dcomp* usq = (dcomp*) malloc(N * sizeof(dcomp));

  // Compute u in real space (we overwrite usq)
  #pragma omp parallel
  {
    #pragma omp single
    ifft(usq,u_hat,N,N);
  }

  // Square value
  for(int s = 0; s < N; s++)
    usq[s] = 3*real(usq[s])*real(usq[s]);

  // Compute the Fourier transform of 3*u^2
  #pragma omp parallel
  {
    #pragma omp single
    fft(usq,B_hat,N,N);
  }

  // Differentiate in Fourier space
  for(int s = 0; s < N; s++)
    B_hat[s] = ikx[s]*B_hat[s];

  free(usq);
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
               double dt, dcomp* y0, dcomp* ikx, int N,
               dcomp* B_hat, dcomp* B_hatm1){

    // initialize method parameters
    long M = round((tfinal - t0) / dt); // number of time steps to integrate

    // initial solution: use the same memory as y0
    // change in name just for readability
    dcomp* ym1 = y0;

    // initial nonlinearity; store in B_hatm1
    kdv_rhs_hat(y0, B_hatm1, ikx, N);

    // take an Euler step -- write to y.
    // y_next = y0 + dt*(A*y + B(y))
    #pragma omp parallel for schedule(static)
    for(int j=0; j<N; j++)
      y[j] = y0[j] + dt*( -ikx[j]*ikx[j]*ikx[j]*y0[j] + B_hatm1[j] );

    // main loop; time steps
    for(int m=0; m<M; m++) {

      printf("Iteration %d/%d\r", m+1,M);

      // compute the nonlinear component; store in B_hat
      // done using OpenMP parallel (I)FFT
      kdv_rhs_hat(y, B_hat, ikx, N);

      // take one update step; update each of the N components
      #pragma omp parallel for schedule(static)
      for(int j=0; j<N; j++) {

        // make a copy of the current iterate
        // needed for parsimonious memory management at the update step
        dcomp yj_temp = y[j];

        // using the TRBDF2 method iteration
        // debugging printouts
        // printf("y[%d] = %f\n", j, y[j]);
        // printf("ym1[j] = %f\n", ym1[j]);
        // printf("B_hat[j] = %f\n", B_hat[j]);
        // printf("B_hatm1[j] = %f\n", B_hatm1[j]);
        y[j] = 4.0*y[j] - ym1[j] + 2.0*dt*(2.0*B_hat[j] - B_hatm1[j]);
        y[j] = y[j] / 3.0;
        y[j] = y[j] / ( 1.0 + 2.0*ikx[j]*ikx[j]*ikx[j]*dt/3.0 ); // simplified a double negative

        // update previous solution and nonlinearity
        ym1[j] = yj_temp;
        B_hatm1[j] = B_hat[j];

      } // component for

    } // time step for

}

int main(int argc, char** argv){

  double t0 = 0; // initial time
  double tfinal = pi; // terminal time
  double dt = 0.01; // fine time step
  printf("t0 = %f\n", t0);
  printf("tfinal = %f\n", tfinal);
  printf("dt = %f\n", dt);

  int N = 64; // number of modes; power of 2
  depth = (int) log2(omp_get_max_threads());

  double L = 60; //size of domain
  double h = L/N; // step size

  printf("Using %d threads...\n",(int) pow(2.0,depth));


  // Preallocate arrays
  dcomp* u       = (dcomp*) malloc(N * sizeof(dcomp)); // numerical ODE solution
  dcomp* u_hat   = (dcomp*) malloc(N * sizeof(dcomp)); //Fourier coefficients
  dcomp* u_hatm1 = (dcomp*) malloc(N * sizeof(dcomp)); // solution at previous time step
  dcomp* u_hat0  = (dcomp*) malloc(N * sizeof(dcomp)); //Fourier coefficients
  dcomp* u_true  = (dcomp*) malloc(N * sizeof(dcomp)); //reference solution
  dcomp* u_hat_true = (dcomp*) malloc(N * sizeof(dcomp)); //reference in Fourier space solution
  dcomp* B_hat   = (dcomp*) malloc(N * sizeof(dcomp)); //nonlinear part of KdV
  dcomp* B_hatm1 = (dcomp*) malloc(N * sizeof(dcomp)); //Fourier coefficients
  dcomp* ikx     = (dcomp*) malloc(N * sizeof(dcomp)); //Fourier modes
  double* xx     = (double*) malloc(N * sizeof(double)); //grid

  /*
  Test problem -- a soliton wave
  */
  // test function u(x,t) = -(c/2) * sech^2( sqrt(c)/2 * (x-c*t) )
  dcomp c = 1.0;
  dcomp arg;

  for(int s = 0; s < N; s++){

    xx[s] = (L/N)*s; //grid point values on [0,L]

    // initial data
    arg = sqrt(c)/2.0 * (xx[s] - c*t0);
    u[s] = -(c/2.0) * pow(cosh(arg), -2.0); //assign function values (memory address u is a placeholder)

    // the true solution at time tfinal
    arg = sqrt(c)/2.0 * (xx[s] - c*tfinal);
    u_true[s] = -(c/2.0) * pow(cosh(arg), -2.0);

    if(s < N/2) ikx[s] = 2*pi*i/L*((dcomp) s); //modes 0,...,N/2-1
    else ikx[s] = 2*pi*i/L*((dcomp) s - (dcomp) N); //modes -N/2,..,-1

  }
  for(int s = 0; s < N; s++) printf("u0[%d] = %f\n", s, u[s]);
  fft(u, u_hat0, N, N); // solve the ODE in Fourier space. Put initial data there.
  fft(u_true, u_hat_true, N, N); // True solution in Fourier space

  // compute the numerical ODE solution in Fourier space
  double tt = omp_get_wtime();
  SBDF2_kdv(u_hat, t0, tfinal, dt, u_hat0, ikx, N, B_hat, B_hatm1);
  printf("Elapsed time = %1.4e\n",omp_get_wtime() - tt);

  // recover the solution in real space via ifft
  ifft(u, u_hat, N, N);

  // debugging printouts
  // for(int s = 0; s < N; s++) printf("u_hat[%d] = %f\n", s, u_hat[s]);
  // for(int s = 0; s < N; s++) printf("u_hat_true[%d] = %f\n", s, u_hat_true[s]);

  // compute error
  double err = 0.0;
  for(int s = 0; s < N; s++)
    err += h*abs( real(u[s]) - real(u_true[s]) );
  printf("SBDF2 Error: %1.10e\n", err);

  double err2 = 0.0;
  for(int s = 0; s < N; s++) {
    err2 += h*abs( real(u_hat[s]) - real(u_hat_true[s]) );
    err2 += h*abs( imag(u_hat[s]) - imag(u_hat_true[s]) );
  }
  printf("SBDF2 Fourier Error: %1.10e\n", err2);

  // tidy
  free(u);
  free(u_hat);
  free(u_hat0);
  free(u_true);
  free(B_hat);
  free(B_hatm1);
  free(ikx);
  free(xx);

  return 0;
}
