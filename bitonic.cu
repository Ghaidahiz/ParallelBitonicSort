#include <cuda_runtime.h>
#include <device_launch_parameters.h> 
#include <iostream> 
#include <vector>
using namespace std;


//
__global__ void sortParallel(int* arr, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
        arr[idx] = arr[idx]; //temp placeholder
}


int main()
{
    //the following randomly generated arrays (1,2,3) and lengths are for the report, while N and numbers is for the demo

    int N1 = 1 << 10;   
    int N2 = 1 << 15;   
    int N3 = 1 << 20;   

    vector<int> numbers1(N1); 
    vector<int> numbers2(N2); 
    vector<int> numbers3(N3); 

    for (int i=0; i<N1; i++){
      numbers1[i]=rand();
    }

    for (int i=0; i<N2; i++){
      numbers2[i]=rand();
    }

    for (int i=0; i<N3; i++){
      numbers3[i]=rand();
    }


    int N=8;

    vector<int> numbers(N); 

    cout << "Enter 8 numbers to sort:\n";
    for (int i = 0; i < N; i++) {
        cin >> numbers[i];
    }
    printf("unsorted list:");
    for (int x : numbers)
    cout << x << " ";


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
    
    
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    int blocks1  = (N1 + threads - 1) / threads;
    int blocks2  = (N2 + threads - 1) / threads;
    int blocks3  = (N3 + threads - 1) / threads;

    sortParallel<<<blocks, threads>>>(d_vec,N);
    sortParallel<<<blocks1, threads>>>(d_vec1,N1);
    sortParallel<<<blocks2, threads>>>(d_vec2,N2);
    sortParallel<<<blocks3, threads>>>(d_vec3,N3);

    cudaError_t err = cudaGetLastError();
    printf("Error: %s\n", cudaGetErrorString(err));

    cudaDeviceSynchronize();



    cudaMemcpy(numbers.data(), d_vec, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(numbers1.data(), d_vec1, N1 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(numbers2.data(), d_vec2, N2 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(numbers3.data(), d_vec3, N3 * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_vec);
    cudaFree(d_vec1);
    cudaFree(d_vec2);
    cudaFree(d_vec3);

    printf("sorted list:");
    for (int x : numbers)
    cout << x << " ";


}
