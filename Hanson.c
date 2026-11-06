/*File Name: jacobi_cond.c
Student's Name: Lu Zhiyuan
Student Number: 3036127517
Development Platform: VScode + Linux (workbench2)
Remark: Using pthread's threads, mutexes, and condition variables to implement a multithreaded Jacobi iterative method for computation.
AI tool (copilot) is used for debugging code and improving format, for example, mutex and cond variable interface. Also, AI tools are used for translating English.
*/

#define _GNU_SOURCE

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <limits.h>              /* For timing */
#include <sys/time.h>            /* For timing */
#include <sys/resource.h>
#include <string.h>
#include <pthread.h>
#include <stdbool.h>
#include <semaphore.h>


/****************Global****************************/

#define MAX(a,b) ((a)>(b)?(a):(b))
#define EPSILON 0.001            /* Termination condition */

char filename[255];                  /* File name of output file */

/* Grid size */
int M = 200;                     /* Number of rows */
int N = 200;                     /* Number of cols */
long max_its = 1000000;          /* Maximum iterations, a safe bound to avoid infinite loop */
double final_diff;               /* Temperature difference between iterations at the end */

/* Thread count */
int thr_count = 2;

/* shared variables between threads */
/*************************************************************/
double** u;                   /* Previous temperatures */
double** w;                   /* New temperatures */


// (1) Add your variables here
typedef struct {
    int id;
    int start_row;
    int end_row;
    double local_diff;
    struct rusage usage;
} thread_data_t;

pthread_t *threads;
thread_data_t *thr_data;

// --- Synchronization Primitives ---
pthread_mutex_t barrier_mutex;
pthread_cond_t cond_master_done;
pthread_cond_t cond_workers_done;

int arrived_workers;
bool terminate;

// Generation counter to prevent barrier reuse race condition
long epoch;
/**************************************************************/

int main (int argc, char *argv[])
{
   int      its;                 /* Iterations to converge */
   double   elapsed;             /* Execution time */
   struct timeval stime, etime;  /* Start and end times */
   struct rusage usage;

   void allocate_2d_array (int, int, double ***);
   void initialize_array (double ***);
   void print_solution (char *, double **);
   int  find_steady_state (void);

   /* For convenience of other problem size testing */
   if ((argc == 1) || (argc == 4)) {
      if (argc == 4) {
         M = atoi(argv[1]);
         N = atoi(argv[2]);
         thr_count = atoi(argv[3]);
      } // Otherwise use default grid and thread size
   } else {
     printf("Usage: %s [ <rows> <cols> <threads ]>\n", argv[0]);
     exit(-1);
   }

   printf("Problem size: M=%d, N=%d\nThread count: T=%d\n", M, N, thr_count);

   /* Create the output file */
   sprintf(filename, "%s_%d_%d_%d.dat", argv[0], M, N, thr_count);

   allocate_2d_array (M, N, &u);
   allocate_2d_array (M, N, &w);
   initialize_array (&u);
   initialize_array (&w);

   gettimeofday (&stime, NULL);
   its = find_steady_state();
   gettimeofday (&etime, NULL);

   elapsed = ((etime.tv_sec*1000000+etime.tv_usec)-(stime.tv_sec*1000000+stime.tv_usec))/1000000.0;

   printf("Converged after %d iterations with error: %8.6f.\n", its, final_diff);
   printf("Elapsed time = %8.4f sec.\n", elapsed);

   /* get the resource usage of the whole process */
   getrusage(RUSAGE_SELF, &usage);
   printf("Program completed - user: %.4f s, system: %.4f s\n",
      (usage.ru_utime.tv_sec + usage.ru_utime.tv_usec/1000000.0),
    (usage.ru_stime.tv_sec + usage.ru_stime.tv_usec/1000000.0));
   printf("no. of context switches: vol %ld, invol %ld\n\n",
  		  usage.ru_nvcsw, usage.ru_nivcsw);

   print_solution (filename, w);
}

/* Allocate two-dimensional array. */
void allocate_2d_array (int r, int c, double ***a)
{
   double *storage;
   int     i;
   storage = (double *) malloc (r * c * sizeof(double));
   *a = (double **) malloc (r * sizeof(double *));
   for (i = 0; i < r; i++)
      (*a)[i] = &storage[i * c];
}

/* Set initial and boundary conditions */
void initialize_array (double ***u)
{
   int i, j;

   /* Set initial values and boundary conditions */
   for (i = 0; i < M; i++) {
      for (j = 0; j < N; j++)
         (*u)[i][j] = 25.0;      /* Room temperature */
      (*u)[i][0] = 0.0;
      (*u)[i][N-1] = 0.0;
   }

   for (j = 0; j < N; j++) {
      (*u)[0][j] = 0.0;
      (*u)[M-1][j] = 1000.0;     /* Heat source */
   }
}

/* Print solution to standard output or a file */
void print_solution (char *filename, double **u)
{
   int i, j;
   char sep;
   FILE *outfile;

   if (!filename) { /* if no filename specified, print on screen */
      sep = '\t';   /* tab added for easier view */
      outfile = stdout;
   } else {
      sep = '\n';   /* for gnuplot format */
      outfile = fopen(filename,"w");
      if (outfile == NULL) {
         printf("Can't open output file.");
         exit(-1);
      }
   }

   /* Print the solution array */
   for (i = 0; i < M; i++) {
      for (j = 0; j < N; j++)
         fprintf (outfile, "%6.2f%c", u[i][j], sep);
      fprintf(outfile, "\n"); /* Empty line for gnuplot */
   }
   if (outfile != stdout)
      fclose(outfile);

}

/* Entry function of the worker threads */
void *thr_func(void *arg) {

// (2) Add the worker's logic here
    thread_data_t *data = (thread_data_t *)arg;
    int i, j;
    int start_row = data->start_row;
    int end_row = data->end_row;
    long my_epoch = 0; // Worker's local epoch counter

    while (true) {
        // --- Computation Phase ---
        data->local_diff = 0.0;
        for (i = start_row; i < end_row; i++) {
            for (j = 1; j < N - 1; j++) {
                w[i][j] = 0.25 * (u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1]);
                if (fabs(w[i][j] - u[i][j]) > data->local_diff) {
                    data->local_diff = fabs(w[i][j] - u[i][j]);
                }
            }
        }

        // --- Barrier Synchronization ---
        pthread_mutex_lock(&barrier_mutex);
        arrived_workers++;

        if (arrived_workers == thr_count) {
            pthread_cond_signal(&cond_workers_done);
        }

        // Wait until the master increments the global epoch.
        // This prevents the barrier reuse race condition.
        while (epoch == my_epoch) {
            pthread_cond_wait(&cond_master_done, &barrier_mutex);
        }
        pthread_mutex_unlock(&barrier_mutex);

        if (terminate) {
            break;
        }
        my_epoch++; // Prepare for the next iteration/epoch
    }

    getrusage(RUSAGE_THREAD, &(data->usage));
    pthread_exit(NULL);
}


int find_steady_state (void)
{

// (3) Implement the thread creation and the main control logic here
   double diff;
   int its;
   int i;
   double **temp;
   struct rusage master_usage;

   // --- Initialization ---
   pthread_mutex_init(&barrier_mutex, NULL);
   pthread_cond_init(&cond_workers_done, NULL);
   pthread_cond_init(&cond_master_done, NULL);
   arrived_workers = 0;
   terminate = false;
   epoch = 0; // Initialize epoch

   threads = (pthread_t *) malloc(thr_count * sizeof(pthread_t));
   thr_data = (thread_data_t *) malloc(thr_count * sizeof(thread_data_t));

   int rows_per_thread = (M - 2) / thr_count;
   int remainder_rows = (M - 2) % thr_count;
   int current_row = 1;

   for (i = 0; i < thr_count; i++) {
      thr_data[i].id = i;
      thr_data[i].start_row = current_row;
      int num_rows = rows_per_thread + (i < remainder_rows ? 1 : 0);
      thr_data[i].end_row = current_row + num_rows;
      current_row = thr_data[i].end_row;
      pthread_create(&threads[i], NULL, thr_func, (void *)&thr_data[i]);
   }

   // --- Main Control Loop (Master Thread) ---
   for (its = 1; its <= max_its; its++) {
      pthread_mutex_lock(&barrier_mutex);
      while (arrived_workers < thr_count) {
         pthread_cond_wait(&cond_workers_done, &barrier_mutex);
      }

      diff = 0.0;
      for (i = 0; i < thr_count; i++) {
         if (thr_data[i].local_diff > diff) {
            diff = thr_data[i].local_diff;
         }
      }
      final_diff = diff;

      // Check for termination for any reason (convergence or max iterations reached).
      if (diff <= EPSILON || its == max_its) {
         terminate = true;
      }

      // Only swap matrices if we are continuing.
      if (!terminate) {
        temp = u;
        u = w;
        w = temp;
      }

      // Reset worker count and advance the epoch to release workers for the next phase.
      arrived_workers = 0;
      epoch++;
      pthread_cond_broadcast(&cond_master_done);
      pthread_mutex_unlock(&barrier_mutex);

      // Exit the loop if termination condition was met.
      if (terminate) {
         break;
      }
   }

   // --- Cleanup and Finalization ---
   for (i = 0; i < thr_count; i++) {
      pthread_join(threads[i], NULL);
      printf("Thread %d has completed - user: %.4f s, system: %.4f s\n",
             thr_data[i].id,
             (thr_data[i].usage.ru_utime.tv_sec + thr_data[i].usage.ru_utime.tv_usec/1000000.0),
             (thr_data[i].usage.ru_stime.tv_sec + thr_data[i].usage.ru_stime.tv_usec/1000000.0));
   }

   getrusage(RUSAGE_THREAD, &master_usage);
   printf("find_steady_state - user: %.4f s, system: %.4f s\n",
      (master_usage.ru_utime.tv_sec + master_usage.ru_utime.tv_usec/1000000.0),
      (master_usage.ru_stime.tv_sec + master_usage.ru_stime.tv_usec/1000000.0));

   pthread_mutex_destroy(&barrier_mutex);
   pthread_cond_destroy(&cond_workers_done);
   pthread_cond_destroy(&cond_master_done);
   free(threads);
   free(thr_data);

   return its;
}
