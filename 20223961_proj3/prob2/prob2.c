#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

#define ARRAY_SIZE 100000000

long long divide_conquer_sum(int* arr, int start, int end, int depth) {
   if (start == end) {
       return arr[start];
   }

   if (depth > 10 || end - start < 10000) {
       long long sum = 0;
       for (int i = start; i <= end; i++) {
           sum += arr[i];
       }
       return sum;
   }

   int mid = (start + end) / 2;
   long long left_sum = 0, right_sum = 0;

#pragma omp parallel sections
   {
#pragma omp section
       {
           left_sum = divide_conquer_sum(arr, start, mid, depth + 1);
       }
#pragma omp section
       {
           right_sum = divide_conquer_sum(arr, mid + 1, end, depth + 1);
       }
   }

   return left_sum + right_sum;
}

int main() {
   int* array;
   long long total_sum;

   array = (int*)malloc(ARRAY_SIZE * sizeof(int));
   if (array == NULL) {
       printf("Memory allocation failed!\n");
       return 1;
   }

   srand(42);
   for (int i = 0; i < ARRAY_SIZE; i++) {
       array[i] = rand() % 100 + 1;
   }

   total_sum = divide_conquer_sum(array, 0, ARRAY_SIZE - 1, 0);

   printf("Array sum: %lld\n", total_sum);

   free(array);

   return 0;
}