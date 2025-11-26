
#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <chrono>

using namespace std;

const int ASCENDING = 1;

void compareAndSwap(vector<int>& a, int i, int j, int dir) {
    if (dir ==  (a[i] > a[j]))  //if dir=1 (ascending) and the first element bigger than the second one swap
        swap(a[i], a[j]);       //if dir=0 (descending)and the first element is not bigger than the second one swap
}


void bitonicMerge(vector<int>& a, int low, int cnt, int dir) {
    if (cnt > 1) {
        int k = cnt / 2;
        for (int i = low; i < low + k; i++)
            compareAndSwap(a, i, i + k, dir);
        bitonicMerge(a, low, k, dir);
        bitonicMerge(a, low + k, k, dir);
    }
}


void bitonicSort(vector<int>& a, int low, int cnt, int dir) {
    if (cnt > 1) {
        int k = cnt / 2;
        // sort the first half in ascending order
        bitonicSort(a, low, k, 1);
        // sort the second half in descending order
        bitonicSort(a, low + k, k, 0);
        // here the comparison and swapping then merging  
        bitonicMerge(a, low, cnt, dir);
    }
}


// function for initilize the array with random numbers from 0 to 1000
vector<int> init(int power){
  int n = pow(2, power);
  vector<int> a(n);
  for (int i = 0; i < n; i++)
      a[i] = rand() % 1000;
  return a;
}

//for printing the array
void print(const vector<int>& a){
  for (int i = 0; i < a.size(); i++)
      cout << a[i] << " ";
  cout << endl;
}

void sort ( vector<int>& a, int power){
    auto start = chrono::high_resolution_clock::now();    // to store the current time befor calling the Bitonic sort
    bitonicSort(a, 0, a.size(), ASCENDING); 
    auto end = chrono::high_resolution_clock::now(); // to store the current time after finishing the sort
    chrono::duration<double, milli> duration = end - start; // calculate the time of sorting
    cout << "Sorting 2^"<<power<<" took " <<duration.count() << " ms" << endl;    
}


int main() {

    int n;
    cout << "Enter the value of n, where 2^n is the size of the array.\nn=  ";
    cin >> n ;

    // initilize the array with random numbers from 0 to 1000
    vector<int> arr = init(n);
    vector<int> arr10=init(10);
    vector<int> arr15=init(15);
    vector<int> arr20=init(20);
   
    //sort and calculate time for user array
    print(arr); // later for the Demo
    sort(arr,n);
    print(arr); // later for the Demo

    //sort and calculate time for 2^10 array 
    sort(arr10,10);

    //sort and calculate time for 2^15 array 
    sort(arr15,15);
    
    //sort and calculate time for 2^20 array 
    sort(arr20,20);
    
    return 0;
}



