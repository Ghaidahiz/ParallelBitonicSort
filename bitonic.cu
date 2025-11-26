#include <cuda_runtime.h>
#include <device_launch_parameters.h> 
#include <iostream> 
#include <vector>
#include <cstdlib>
using namespace std;

__global__ void bitonicSort(int* arr, int N, int j, int k)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int ixj = idx ^ j;

    if (ixj > idx && ixj < N)
    {
        // ascending sort if idx & k == 0, descending otherwise
        if ((idx & k) == 0)
        {
            if (arr[idx] > arr[ixj])
            {
                int temp = arr[idx];
                arr[idx] = arr[ixj];
                arr[ixj] = temp;
            }
        }
        else
        {
            if (arr[idx] < arr[ixj])
            {
                int temp = arr[idx];
                arr[idx] = arr[ixj];
                arr[ixj] = temp;
            }
        }
    }
}

void sortOnGPU(int* d_arr, int N)
{
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    for (int k = 2; k <= N; k <<= 1)       // stage size
    {
        for (int j = k >> 1; j > 0; j >>= 1)   // step
        {
            bitonicSort<<<blocks, threads>>>(d_arr, N, j, k);
            cudaDeviceSynchronize();
        }
    }
}

int main()
{
    int N1 = 1 << 10;   
    int N2 = 1 << 15;   
    int N3 = 1 << 20;   

    vector<int> numbers1(N1); 
    vector<int> numbers2(N2); 
    vector<int> numbers3(N3); 

    for (int i=0; i<N1; i++) numbers1[i] = rand();
    for (int i=0; i<N2; i++) numbers2[i] = rand();
    for (int i=0; i<N3; i++) numbers3[i] = rand();
       
    int N=8;
    vector<int> numbers(N); 
    cout << "Enter 8 numbers to sort them:\n";
    for (int i = 0; i < N; i++) cin >> numbers[i];

    int* d_vec;    
    int* d_vec1;
    int* d_vec2;
    int* d_vec3;

    cudaMalloc(&d_vec, N * sizeof(int));
    cudaMalloc(&d_vec1, N1 * sizeof(int));    
    cudaMalloc(&d_vec2, N2 * sizeof(int));
    cudaMalloc(&d_vec3, N3 * sizeof(int));

    cudaMemcpy(d_vec, numbers.data(), N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec1, numbers1.data(), N1 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec2, numbers2.data(), N2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec3, numbers3.data(), N3 * sizeof(int), cudaMemcpyHostToDevice);

    //to calculate time taken in each operation
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds;

    //sorts user input
    cudaEventRecord(start);
    sortOnGPU(d_vec, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Sorting user input (N=" << N << ") took " << milliseconds << " ms\n";

    //sorts 2^10 random numbers
    cudaEventRecord(start);
    sortOnGPU(d_vec1, N1);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Sorting 2^10 random numbers=" << N1 << " took " << milliseconds << " ms\n";

    //sorts 2^15 random numbers
    cudaEventRecord(start);
    sortOnGPU(d_vec2, N2);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Sorting 2^15 random numbers=" << N2 << " took " << milliseconds << " ms\n";

    //sorts 2^20 random numbers
    cudaEventRecord(start);
    sortOnGPU(d_vec3, N3);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Sorting 2^20 random numbers=" << N3 << " took " << milliseconds << " ms\n";

    cudaMemcpy(numbers.data(), d_vec, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(numbers1.data(), d_vec1, N1 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(numbers2.data(), d_vec2, N2 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(numbers3.data(), d_vec3, N3 * sizeof(int), cudaMemcpyDeviceToHost);

    cout << "Sorted numbers:\n";
    for (int x : numbers) cout << x << " ";
    cout << endl;

    //cleanup
    cudaFree(d_vec);
    cudaFree(d_vec1);
    cudaFree(d_vec2);
    cudaFree(d_vec3);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
