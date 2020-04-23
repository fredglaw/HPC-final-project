/* OpenMP Implementation of the FFT
 * The general framework uses tasks:
 *   #pragma omp parallel
 *   {
 *     #pragma omp single
 *     fft(x,x_hat,N,N);
 *   }
 * Call code as 
 *  ./fft (N) (depth,optional)
 * depth is the level of parallelism in recursion num_threads is 2^depth
 * Compile as
 *   g++ -std=c++11 -O3 -march=native -fopenmp -o fft fft.cpp
*/

#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <complex>
#include <fftw3.h>

using namespace std ;
typedef complex<double> dcomp;

//int depth = 2.0; // 2^depth threads will be used
int depth;
const double pi = 4.0*atan(1); 
const dcomp i = sqrt((dcomp) -1); //unit imaginary

dcomp complex_exp_taylor(const double x){

  double C1 = 1.0;
  double C2 = x;
  double val1 = 1.0;
  double val2 = x;
  int s = 1.0; 
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
void fft(const dcomp* x,dcomp* x_hat, const int N, const int Nmodes){

  if(N == 1) x_hat[0] = x[0];
  else{

    int Nmod2 = N/2;

    // Preallocate local x_hat array
    dcomp* x_hat_loc = (dcomp*) malloc(N * sizeof(dcomp)); 
    
    for(int s = 0; s < Nmod2; s++){
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
    for(int s = 0; s < Nmod2; s++){
      t = x_hat_loc[s];
      xnhat = C*x_hat_loc[s+Nmod2];
      x_hat_loc[s] = t + xnhat;    
      x_hat_loc[s+Nmod2] = t - xnhat; 
      C = C*w;   
    }
   
    for(int s = 0; s < N; s++) x_hat[s] = x_hat_loc[s];    
    free(x_hat_loc);

  } 
}

/* Computes the inverse DFT of x_hat
 * x: ouput array of grid values
 * x_hat: input array of Fourier coefficients
 * N: size of current array (changes in recursion)
 * Nmodes: size of full array
*/
void ifft(dcomp* x,const dcomp* x_hat,const int N,const int Nmodes){

  if(N == 1) x[0] = x_hat[0]/((dcomp) Nmodes);
  else{
    
    int Nmod2 = N/2;
    
    // Preallocate local x array
    dcomp* x_loc = (dcomp*) malloc(N * sizeof(dcomp));
 
    for(int s = 0; s < Nmod2; s++){
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
    for(int s = 0; s < Nmod2; s++){
      t = x_loc[s];
      xn = C*x_loc[s+Nmod2];
      x_loc[s] = t + xn;    
      x_loc[s+Nmod2] = t - xn;    
      C = C*w;
    }
    
    // Write sums to full array
    for(int s = 0; s < N; s++) x[s] = x_loc[s];
    free(x_loc);

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

// Main function
int main(int argc, char** argv) {

  int N; //number of modes, input from command line
  double L = 2*pi; //size of domain

  /*** Input conditions ***/
  if(argc < 3){
    if(argc == 2){N = atoi(argv[1]); depth = 0.0;}
    else{
      printf("\n Call instructions:\n    ./fft (N) (depth, optional) \n\n If depth is not included, code runs in serial.\n\n"); 
      return 0;
    }
  }
  else{N = atoi(argv[1]); depth = atoi(argv[2]);}

  if(pow(2,depth) > omp_get_max_threads())
    depth = (int) log2(omp_get_max_threads());
    
  // Check if N is a power of 2
  int check = N;
  while(check > 1){ 
    if(check % 2 != 0){
      printf("Error: N must be power of 2.\n");
      return 0;
    }
    check = check/2;
  }

  printf("Using %d threads...\n",(int) pow(2.0,depth));
  
  // Preallocate arrays 
  dcomp* u = (dcomp*) malloc(N * sizeof(dcomp)); //solution vector
  dcomp* u_hat = (dcomp*) malloc(N * sizeof(dcomp)); //Fourier coefficients
  dcomp* B = (dcomp*) malloc(N * sizeof(dcomp)); //nonlinear part of KdV
  dcomp* B_hat = (dcomp*) malloc(N * sizeof(dcomp)); //Fourier coefficients
  dcomp* kdv_ref = (dcomp*) malloc(N * sizeof(dcomp)); //reference solution

  dcomp* ikx = (dcomp*) malloc(N * sizeof(dcomp)); //Fourier modes
  double* xx = (double*) malloc(N * sizeof(double)); //grid

// test function u(x) = exp(-c*(x-L/2)^2)
// d/dx(3 * u^2) = -12*c*(x - L/2)*e^(-2*c*(x - L/2)^2
  dcomp c = 10.0;
  dcomp A = 1.0;
  dcomp arg;
 
  for(int s = 0; s < N; s++){
    xx[s] = (L/N)*s; //grid point values on [0,L] 
    arg = -c*pow(xx[s] - L/2,2);
    u[s] = A*exp(arg); //assign function values

    kdv_ref[s] = 3.0*(-4.0*c*(xx[s] - L/2))*u[s]*u[s];

    if(s < N/2) ikx[s] = 2*pi*i/L*((dcomp) s); //modes 0,...,N/2-1
    else ikx[s] = 2*pi*i/L*((dcomp) s - (dcomp) N); //modes -N/2,..,-1

  }

 
  /*** Test for KdV equation ***/
  double tt = omp_get_wtime();
  fft(u,u_hat,N,N); 
  printf("Elapsed time = %1.4e\n",omp_get_wtime() - tt);
    
  // Check solution
  kdv_rhs_hat(u_hat,B_hat,ikx,N); 
  ifft(B,B_hat,N,N);

  double err = 0.0;
  for(int s = 0; s < N; s++) err += abs(real(B[s]) - real(kdv_ref[s]));
  printf("Error in KdV operator = %1.4e\n",err);

  /*** FFTW COMPARISON ***/
  fftw_complex *in = (fftw_complex*) fftw_malloc(N*sizeof(fftw_complex));
  fftw_complex *out = (fftw_complex*) fftw_malloc(N*sizeof(fftw_complex)); 
  fftw_plan p = fftw_plan_dft_1d(N,in,out,1,FFTW_ESTIMATE);

  for(int s = 0; s < N; s++){
    double val = 1.0*exp(-10*pow(xx[s] - L/2,2));
    in[s][0] = val; in[s][1] = 0.0;
  }

  tt = omp_get_wtime();
  fftw_execute(p);
  printf("Elapsed time = %1.4e\n",omp_get_wtime() - tt);

  err = 0.0;
  for(int s = 0; s < N; s++) err += abs(real(u_hat[s]) - out[s][0])/N;
  printf("Error = %1.4e\n",err);

  //Clean up
  fftw_destroy_plan(p);
  fftw_free(in); fftw_free(out);
  free(u); free(u_hat);
  free(B); free(B_hat);
  free(kdv_ref);
  free(ikx);
  return 0;

}

