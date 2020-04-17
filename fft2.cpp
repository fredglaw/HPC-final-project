#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <complex>

using namespace std ;
typedef complex<double> dcomp;

int n_threads;
double pi = 4.0*atan(1); 
dcomp i = sqrt((dcomp) -1); //unit imaginary

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
    fft(x1,x1_hat,N/2,Nmodes);   
    fft(x2,x2_hat,N/2,Nmodes);
    
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
 * x: output array of grid values
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
    ifft(x1,x1_hat,N/2,Nmodes);   
    ifft(x2,x2_hat,N/2,Nmodes);
    
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

/* Two-dimensional parallel FFT
 *  This function consecutively computes the FFT in the x-direction and then
 *  the y-direction. 
 */
void fft2(dcomp* u, dcomp* u_hat, int Nx, int Ny){

  //fft in x-direction
  #pragma omp parallel for num_threads(n_threads) schedule(static)
  for(int sy = 0; sy < Ny; sy++)
    fft(u + Nx*sy,u_hat + Nx*sy,Nx,Nx);
  
  //fft in y-direction
  #pragma omp parallel for num_threads(n_threads) schedule(static)
  for(int sx = 0; sx < Nx; sx++){
    dcomp* u_hat_loc = (dcomp*) malloc(Ny * sizeof(dcomp));
    for(int sy = 0; sy < Ny; sy++)
      u_hat_loc[sy] = u_hat[sx + Nx*sy];
    fft(u_hat_loc,u_hat_loc,Ny,Ny);
    for(int sy = 0; sy < Ny; sy++) 
      u_hat[sx + Nx*sy] = u_hat_loc[sy];
    free(u_hat_loc);
  }

}

/* Two-dimensional parallel IFFT
 * This function consecutievly computes the IFFT in the x-drection and then
 * the y-direction.
 */
void ifft2(dcomp* u, dcomp* u_hat, int Nx, int Ny){

  //fft in x-direction
  #pragma omp parallel for num_threads(n_threads) schedule(static)
  for(int sy = 0; sy < Ny; sy++)
    ifft(u + Nx*sy,u_hat + Nx*sy,Nx,Nx);
  
  //fft in y-direction
  #pragma omp parallel for num_threads(n_threads) schedule(static)
  for(int sx = 0; sx < Nx; sx++){
    dcomp* u_loc = (dcomp*) malloc(Ny * sizeof(dcomp));
    for(int sy = 0; sy < Ny; sy++)
      u_loc[sy] = u[sx + Nx*sy];
    ifft(u_loc,u_loc,Ny,Ny);
    for(int sy = 0; sy < Ny; sy++) 
      u[sx + Nx*sy] = u_loc[sy];
    free(u_loc);
  }

}


// Main function
int main(int argc, char** argv) {

  int Nx, Ny, N; //number of modes, input from command line
  double Lx = 2*pi; //size of domain
  double Ly = 2*pi;

  /*** Input conditions ***/
  if(argc < 3){
    printf("\n Call instructions:\n    ./fft2 (N) (n_threads)\n\n");
    return 0;
  }
  else{
    N = atoi(argv[1]);
    n_threads = atoi(argv[2]);
  }

  // Check if N is a power of 2
  int check = N;
  while(check > 1){ 
    if(check % 2 != 0){
      printf("Error: N must be power of 2.\n");
      return 0;
    }
    check = check/2;
  }

  Nx = N; Ny = N;
  int  mem_size = Nx*Ny*sizeof(dcomp);
 
  // Preallocate arrays 
  dcomp* u = (dcomp*) malloc(mem_size); //solution vector
  dcomp* v = (dcomp*) malloc(mem_size); //test solution
  dcomp* u_hat = (dcomp*) malloc(mem_size); //Fourier coefficients
  dcomp* ikx = (dcomp*) malloc(N * sizeof(dcomp)); //x-Fourier modes
  dcomp* iky = (dcomp*) malloc(N * sizeof(dcomp)); //y-Fourier modes
  double* xx = (double*) malloc(N * sizeof(double)); //x-grid
  double* yy = (double*) malloc(N * sizeof(double)); //y-grid

  // Construct x-grid and x-modes
  for(int sx = 0; sx < Nx; sx++){
    xx[sx] = (Lx/Nx)*sx;
    if(sx < Nx/2) ikx[sx] = 2*pi*i/Lx*((dcomp) sx); //modes 0,...,N/2-1
    else ikx[sx] = 2*pi*i/Lx*((dcomp) sx - (dcomp) Nx); //modes -N/2,..,-1
  }  
  // Construct y-grid and y-modes
  for(int sy = 0; sy < Ny; sy++){
    yy[sy] = (Ly/Ny)*sy;
    if(sy < Ny/2) iky[sy] = 2*pi*i/Ly*((dcomp) sy); //modes 0,...,N/2-1
    else iky[sy] = 2*pi*i/Ly*((dcomp) sy - (dcomp) Ny); //modes -N/2,..,-1
  }

  // Fill in u
  for(int sy = 0; sy < Ny; sy++){
    for(int sx = 0; sx < Nx; sx++){
      u[sx + Nx*sy] = sin(2*pi*xx[sx]/Lx)*sin(2*pi*yy[sy]/Ly);
      u_hat[sx + Nx*sy] = 0.0;
    }
  }

  // Check that ifft2(fft2(u))=u and time
  double tt = omp_get_wtime();
  fft2(u,u_hat,Nx,Ny);
  ifft2(v,u_hat,Nx,Ny);
  printf("Time elapsed: %1.4e\n",omp_get_wtime() - tt);
  double err = 0.0;

  #pragma omp parallel for schedule(static) reduction(+:err)
  for(int sx = 0; sx < Nx; sx++){
    for(int sy = 0; sy < Ny; sy++){
    err += abs(u[sx+Nx*sy] - v[sx+Nx*sy])/N;
    }
  }

  printf("Error = %1.4e\n",err);

  //Clean up
  free(u); free(u_hat);
  free(ikx); free(iky);
  free(v);
  return 0;

}

