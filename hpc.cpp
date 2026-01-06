#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>
#define N [ 3, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650, 1700, 1750, 1800, 1850, 1900, 1950, 2000, 3000, 4000, 5000, 6000, 7000, 8000 ]
int main(int argc, char **argv)
{
    //    long long sum = 0;
    // #pragma omp parallel for reduction(+ : sum)
    //    for (long long i = 1; i <= 1000000; i++)
    //    {
    //        sum += i;
    //    }
    //    printf("OpenMP Sum: %lld\n", sum);

    static double A[N][N], B[N][N], C[N][N];

    // Initialize matrices
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
        {
            A[i][j] = 1.0;
            B[i][j] = 1.0;
            C[i][j] = 0.0;
        }
    // Matrix multiplication using OpenMP
#pragma omp parallel for
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            double sum = 0.0;
            for (int k = 0; k < N; k++)
            {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
    printf("C[0][0]: %f\n", C[0][0]);
    return 0;
}
