#include <iostream>
#include <cstdlib>
#include <omp.h>
#include <chrono>
#include <iomanip>
#define BLOCKSIZE 64
#define NBLOCKS 32

using namespace std;

void init_matrix(double *A, double value, int heightA, int widthA)
{
    for (int k = 0; k < heightA * widthA; k++)
    {
        A[k] = value;
    }
}

void dgemm(double alpha, double *A, double *Btranspose, double beta, double *C, int heightA, int widthA)
{
    #pragma omp parallel for simd schedule(dynamic)
    for (int i = 0; i < heightA; i++)
    {
        for (int j = 0; j < widthA; j++)
        {
            double temp = beta * C[widthA*i + j];
            for (int k = 0; k < heightA; k++)
            {
                temp += alpha * A[widthA * i + k] * Btranspose[widthA * j + k];
                // C[widthA*i + j] = A[widthA*i + k] * Btranspose[heightA*k + j];
            }
            C[widthA * i + j] = temp;
        }
    }
}

void dgemm_part(double alpha, double *A, double *Btranspose, double beta, double *C, int iBegin, int jBegin, int kBegin, int blockSize)
{
    for (int i = iBegin; i < iBegin + blockSize; i++)
    {
        for (int j = jBegin; j < jBegin + blockSize; j++)
        {
            double temp = 0.f;
            for (int k = kBegin; k < kBegin + blockSize; k++)
            {
                temp += alpha * A[blockSize * i + k] * Btranspose[blockSize * j + k];
            }
            #pragma omp atomic
            C[blockSize * i + j] += temp;
        }
    }
}

void dgemm_blocks(double alpha, double *A, double *Btranspose, double beta, double *C, int blockHeight, int blockWidth, int blockSize)
{
    #pragma omp parallel for simd
    for (int k = 0; k < blockHeight * blockSize * blockWidth * blockSize; k++)
    {
        C[k] = beta * C[k];
    }
    #pragma omp parallel for simd schedule(dynamic)
    for (int i = 0; i < blockHeight; i++)
    {
        for (int j = 0; j < blockWidth; j++)
        {
            for (int k = 0; k < blockWidth; k++)
            {
                dgemm_part(alpha, A, Btranspose, beta, C, blockSize * i, blockSize * j, blockSize * k, blockSize);
            }
        }
    }
}

double sum(double *A, int height, int width)
{
    double sum = 0.f;
    #pragma omp parallel for simd reduction(+:sum)
    for (int k = 0; k < height * width; k++)
    {
        sum += A[k];
    }
    return sum;
}

int main(int argc, char *argv[])
{
    double *A = new double[BLOCKSIZE * NBLOCKS * BLOCKSIZE * NBLOCKS];
    double *B = new double[BLOCKSIZE * NBLOCKS * BLOCKSIZE * NBLOCKS];
    double *C = new double[BLOCKSIZE * NBLOCKS * BLOCKSIZE * NBLOCKS];

    init_matrix(A, 1.f, BLOCKSIZE * NBLOCKS, BLOCKSIZE * NBLOCKS);
    init_matrix(B, 1.f, BLOCKSIZE * NBLOCKS, BLOCKSIZE * NBLOCKS);
    init_matrix(C, 1.f, BLOCKSIZE * NBLOCKS, BLOCKSIZE * NBLOCKS);


    double alpha = 2.f;
    double beta = 2.f;

    cout << BLOCKSIZE*NBLOCKS << " * " << BLOCKSIZE*NBLOCKS << " DGEMM" << endl;
    cout << "Block size for tiled DGEMM: " << BLOCKSIZE << endl;

    auto start = chrono::steady_clock::now();
    dgemm(alpha, A, B, beta, C, BLOCKSIZE * NBLOCKS, BLOCKSIZE * NBLOCKS);
    auto end = chrono::steady_clock::now();
    cout << "DGEMM time: " << chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.f << " ms" << endl;

    start = chrono::steady_clock::now();
    double result = sum(C, BLOCKSIZE * NBLOCKS, BLOCKSIZE * NBLOCKS);
    end = chrono::steady_clock::now();
    cout << "Sum time: " << chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.f << " ms" << endl;
    cout << "Result: " << fixed << setprecision(1) << result << endl;

    init_matrix(C, 1.f, BLOCKSIZE * NBLOCKS, BLOCKSIZE * NBLOCKS);
    start = chrono::steady_clock::now();
    dgemm_blocks(alpha, A, B, beta, C, NBLOCKS, NBLOCKS, BLOCKSIZE);
    end = chrono::steady_clock::now();
    cout << "DGEMM time (tiled): " << chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.f << " ms" << endl;

    start = chrono::steady_clock::now();
    double result2 = sum(C, BLOCKSIZE * NBLOCKS, BLOCKSIZE * NBLOCKS);
    end = chrono::steady_clock::now();
    cout << "Sum time: " << chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.f << " ms" << endl;
    cout << "Result: " << fixed << setprecision(1) << result2 << endl;

    delete A;
    delete B;
    delete C;

    return 0;
}