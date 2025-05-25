#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// 소수 판별 함수
int is_prime(int n) {
    if (n <= 1) return 0;
    if (n == 2) return 1;
    if (n % 2 == 0) return 0;
    
    for (int i = 3; i <= n; i++) {
        if (n % i == 0)
            return 0;
    }
    return 1;
}
int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("Usage: %s <scheduling_type> <num_threads>\n", argv[0]);
        printf("scheduling_type: 1=static, 2=dynamic, 3=static(10), 4=dynamic(10)\n");
        printf("num_threads: 1,2,4,6,8,10,12,14,16\n");
        return 1;
    }

    int schedule_type = atoi(argv[1]);
    int num_threads = atoi(argv[2]);

    omp_set_num_threads(num_threads);

    int count = 0;
    double start_time, end_time;

    start_time = omp_get_wtime();

    switch (schedule_type) {
    case 1:
#pragma omp parallel for schedule(static) reduction(+:count)
        for (int i = 1; i <= 200000; i++) {
            if (is_prime(i)) {
                count++;
            }
        }
        break;

    case 2:
#pragma omp parallel for schedule(dynamic) reduction(+:count)
        for (int i = 1; i <= 200000; i++) {
            if (is_prime(i)) {
                count++;
            }
        }
        break;

    case 3:
#pragma omp parallel for schedule(static, 10) reduction(+:count)
        for (int i = 1; i <= 200000; i++) {
            if (is_prime(i)) {
                count++;
            }
        }
        break;

    case 4:
#pragma omp parallel for schedule(dynamic, 10) reduction(+:count)
        for (int i = 1; i <= 200000; i++) {
            if (is_prime(i)) {
                count++;
            }
        }
        break;

    default:
        printf("Invalid scheduling type. Use 1, 2, 3, or 4.\n");
        return 1;
    }

    end_time = omp_get_wtime();

    double execution_time_ms = (end_time - start_time) * 1000;

    printf("Prime numbers: %d\n", count);
    printf("Execution time: %.3f ms\n", execution_time_ms);

    return 0;
}