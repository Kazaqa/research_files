#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#define M 8
#define N 8
#define lda M
#define ldb N
#define ldc N
#define IDX2C(i,j,ld) (((j)*(ld))+(i))


int main (void){
    cudaError_t cudaStatA;
    cudaError_t cudaStatB;
    cudaError_t cudaStatC;
    cublasStatus_t stat;
    cublasHandle_t handle; 

    cublasSideMode_t side = CUBLAS_SIDE_LEFT; 
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    cublasOperation_t trans = CUBLAS_OP_N; 
    cublasDiagType_t diag = CUBLAS_DIAG_NON_UNIT;

    int i, j;
    float* devPtrA;
    float* devPtrB;
    float* devPtrC;
    float* alpha = (float *)1;
    float* a = 0;
    float* b = 0;
    float* c = 0;
    a = (float *)malloc (M * M * sizeof (*a));
    b = (float *)malloc (M * N * sizeof (*b));
    c = (float *)malloc (M * N * sizeof (*c));

    if (!a) {
        printf ("host memory allocation failed");
        return EXIT_FAILURE;
    }

    for (j = 0; j < M; j++) {
        for (i = 0; i < M; i++) {
            if(i > j){
                a[IDX2C(i,j,M)] = 0;
            }
            else if(rand()%2==0){
                a[IDX2C(i,j,M)] = rand();
            }
            else{
                a[IDX2C(i,j,M)] = rand()*(-1);
            }
        }
    }

    for (j = 0; j < N; j++) {
        for (i = 0; i < M; i++) {
            if(rand()%2==0){
                b[IDX2C(i,j,M)] = rand();
            }
            else{
                b[IDX2C(i,j,M)] = rand()*(-1);
            }
            c[IDX2C(i,j,M)] = 0;
        }
    }


    cudaStatA = cudaMalloc ((void**)&devPtrA, M*M*sizeof(*a));
    cudaStatB = cudaMalloc ((void**)&devPtrB, M*N*sizeof(*b));
    cudaStatC = cudaMalloc ((void**)&devPtrC, M*N*sizeof(*c));
    if (cudaStatA != cudaSuccess) {
        printf ("device memory allocation failed(A)");
        return EXIT_FAILURE;
    }
    if (cudaStatB != cudaSuccess) {
        printf ("device memory allocation failed(B)");
        return EXIT_FAILURE;
    }
    if (cudaStatC != cudaSuccess) {
        printf ("device memory allocation failed(C)");
        return EXIT_FAILURE;
    }



    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }
    stat = cublasSetMatrix (M, M, sizeof(*a), a, M, devPtrA, M);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download failed(A)");
        cudaFree (devPtrA);
        cudaFree (devPtrB);
        cudaFree (devPtrC);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }
    stat = cublasSetMatrix (M, N, sizeof(*b), b, M, devPtrB, M);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download failed(B)");
        cudaFree (devPtrA);
        cudaFree (devPtrB);
        cudaFree (devPtrC);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }

    stat = cublasStrmm(handle,side,uplo,trans,diag,M,N,alpha,a,lda,b,ldb,c,ldc);
    
    stat = cublasGetMatrix (M, N, sizeof(*c), devPtrC, M, c, M);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data upload failed");
        cudaFree (devPtrA);
        cudaFree (devPtrB);
        cudaFree (devPtrC);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }
    cudaFree (devPtrA);
    cudaFree (devPtrB);
    cudaFree (devPtrC);
    cublasDestroy(handle);
    for (j = 0; j < N; j++) {
        for (i = 0; i < M; i++) {
            printf ("%7.0f", c[IDX2C(i,j,M)]);
        }
        printf ("\n");
    }
    free(a);
    free(b);
    free(c);
    return EXIT_SUCCESS;
}