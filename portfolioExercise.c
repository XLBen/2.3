//
// Starting code for the portfolio exercise. Some required routines are included in a separate
// file (ending '_extra.h'); this file should not be altered, as it will be replaced with a different
// version for assessment.
//
// Compile as normal, e.g.,
//
// > gcc -o portfolioExercise portfolioExercise.c -pthread
//
// and launch with the problem size N and number of threads p as command line arguments, e.g.,
//
// > ./portfolioExercise 12 4
//

//
// Includes.
//
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>

#include "portfolioExercise_extra.h"        // Contains routines not essential to the assessment.

typedef struct {
    float **M;
    float *u;
    float *v;
    int N;
    int nThreads;
    int thread_id;
    float *dotProduct;
    pthread_mutex_t *mutex;
} ThreadArgs;

void* parallel_compute(void* arg) {
    ThreadArgs *args = (ThreadArgs*)arg;

    int N = args->N;
    int nThreads = args->nThreads;
    int thread_id = args->thread_id;
    float **M = args->M;
    float *u = args->u;
    float *v = args->v;

    int rows_per_thread = N / nThreads;
    int start_row = thread_id * rows_per_thread;
    int end_row = start_row + rows_per_thread;

    for (int row = start_row; row < end_row; row++) {
        v[row] = 0.0f;
        for (int col = 0; col < N; col++) {
            v[row] += M[row][col] * u[col];
        }
    }

    float local_dot = 0.0f;
    for (int i = start_row; i < end_row; i++) {
        local_dot += v[i] * v[i];
    }

    pthread_mutex_lock(args->mutex);
    *(args->dotProduct) += local_dot;
    pthread_mutex_unlock(args->mutex);

    return NULL;
}

//
// Main.
//
int main( int argc, char **argv )
{
    //
    // Initialisation and set-up.
    //

    // Get problem size and number of threads from command line arguments.
    int N, nThreads;
    if( parseCmdLineArgs(argc,argv,&N,&nThreads)==-1 ) return EXIT_FAILURE;

    // Initialise (i.e, allocate memory and assign values to) the matrix and the vectors.
    float **M, *u, *v;
    if( initialiseMatrixAndVector(N,&M,&u,&v)==-1 ) return EXIT_FAILURE;

    // For debugging purposes; only display small problems (e.g., N=8 and nThreads=2 or 4).
    if( N<=12 ) displayProblem( N, M, u, v );

    // Start the timing now.
    struct timespec startTime, endTime;
    clock_gettime( CLOCK_REALTIME, &startTime );

    //
    // Parallel operations, timed.
    //
    float dotProduct = 0.0f;        // You should leave the result of your calculation in this variable.

    pthread_mutex_t mutex;
    pthread_mutex_init(&mutex, NULL);

    pthread_t threads[nThreads];
    ThreadArgs args[nThreads];

    for (int i = 0; i < nThreads; i++) {
        args[i].M = M;
        args[i].u = u;
        args[i].v = v;
        args[i].N = N;
        args[i].nThreads = nThreads;
        args[i].thread_id = i;
        args[i].dotProduct = &dotProduct;
        args[i].mutex = &mutex;

        pthread_create(&threads[i], NULL, parallel_compute, &args[i]);
    }

    for (int i = 0; i < nThreads; i++) {
        pthread_join(threads[i], NULL);
    }

    pthread_mutex_destroy(&mutex);

    // After completing Step 1, you can uncomment the following line to display M, u and v, to check your solution so far.
    // if( N<=12 ) displayProblem( N, M, u, v );

    // DO NOT REMOVE OR MODIFY THIS PRINT STATEMENT AS IT IS REQUIRED BY THE ASSESSMENT.
    printf( "Result of parallel calculation: %f\n", dotProduct );

    //
    // Check against the serial calculation.
    //

    // Output final time taken.
    clock_gettime( CLOCK_REALTIME, &endTime );
    double seconds = (double)( endTime.tv_sec + 1e-9*endTime.tv_nsec - startTime.tv_sec - 1e-9*startTime.tv_nsec );
    printf( "Time for parallel calculations: %g secs.\n", seconds );

    // Step 1. Matrix-vector multiplication Mu = v.
    for( int row=0; row<N; row++ )
    {
        v[row] = 0.0f;              // Make sure the right-hand side vector is initially zero.

        for( int col=0; col<N; col++ )
            v[row] += M[row][col] * u[col];
    }

    // Step 2: The dot product of the vector v with itself
    float dotProduct_serial = 0.0f;
    for( int i=0; i<N; i++ ) dotProduct_serial += v[i]*v[i];

    // DO NOT REMOVE OR MODIFY THIS PRINT STATEMENT AS IT IS REQUIRED BY THE ASSESSMENT.
    printf( "Result of the serial calculation: %f\n", dotProduct_serial );

    //
    // Clear up and quit.
    //
    freeMatrixAndVector( N, M, u, v );

    return EXIT_SUCCESS;
}