#define _GNU_SOURCE

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <errno.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <string.h>
#include <pthread.h>
#include <semaphore.h>

#define EPSILON 0.001

double **u;
double **w;
int M = 200;
int N = 200;
int T = 2;
long max_its = 1000000;
double final_diff;

pthread_t *threads;

typedef struct {
    int id;
    int start_row;
    int end_row;
    double local_diff;
    struct rusage usage;
} thread_data_t;

thread_data_t *workers;

sem_t sem_worker_done;
sem_t sem_master_go;

volatile int terminate;

char filename[255];

static void allocate_2d_array(int r, int c, double ***a);
static void initialize_array(double **a, int M, int N);
static void print_solution(const char *filename, double **a, int M, int N);
static inline void compute_row_range(int start, int end, double *out_diff);
static void *worker_func(void *arg);
static void init_solver(void);
static void cleanup_solver(void);
static void print_thread_metrics(const thread_data_t *td);
static int find_steady_state(void);

int main(int argc, char *argv[])
{
    struct timeval stime, etime;
    struct rusage usage;
    double elapsed;
    int its;

    if (argc == 3) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
    } else if (argc == 4) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        T = atoi(argv[3]);
    } else if (argc != 1) {
        printf("Usage: %s [ <rows> <cols> [<threads>] ]\n", argv[0]);
        exit(-1);
    }

    if (M < 3 || N < 3 || T <= 0) {
        printf("Invalid parameters: rows>=3, cols>=3, threads>0 required.\n");
        exit(-1);
    }

    printf("Problem size: M=%d, N=%d\nThread count: T=%d\n", M, N, T);

    sprintf(filename, "jacobi_sema_%d_%d_%d.dat", M, N, T);

    allocate_2d_array(M, N, &u);
    allocate_2d_array(M, N, &w);
    initialize_array(u, M, N);
    initialize_array(w, M, N);

    gettimeofday(&stime, NULL);
    its = find_steady_state();
    gettimeofday(&etime, NULL);

    elapsed = ((etime.tv_sec * 1000000 + etime.tv_usec) -
               (stime.tv_sec * 1000000 + stime.tv_usec)) / 1000000.0;

    printf("Converged after %d iterations with error: %8.6f.\n",
           its, final_diff);
    printf("Elapsed time = %8.4f sec.\n", elapsed);

    getrusage(RUSAGE_SELF, &usage);
    printf("Program completed - user: %.4f s, system: %.4f s\n",
           usage.ru_utime.tv_sec + usage.ru_utime.tv_usec / 1000000.0,
           usage.ru_stime.tv_sec + usage.ru_stime.tv_usec / 1000000.0);
    printf("no. of context switches: vol %ld, invol %ld\n\n",
           usage.ru_nvcsw, usage.ru_nivcsw);

    print_solution(filename, u, M, N);

    return 0;
}

static void allocate_2d_array(int r, int c, double ***a)
{
    double *storage = (double *) malloc(r * c * sizeof(double));
    if (!storage) { perror("malloc"); exit(-1); }

    *a = (double **) malloc(r * sizeof(double *));
    if (!*a) { perror("malloc"); exit(-1); }

    for (int i = 0; i < r; i++)
        (*a)[i] = &storage[i * c];
}

static void initialize_array(double **a, int M, int N)
{
    int i, j;

    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++)
            a[i][j] = 25.0;
        a[i][0] = 0.0;
        a[i][N - 1] = 0.0;
    }

    for (j = 0; j < N; j++) {
        a[0][j] = 0.0;
        a[M - 1][j] = 1000.0;
    }
}

static void print_solution(const char *filename, double **a, int M, int N)
{
    int i, j;
    char sep;
    FILE *outfile;

    if (!filename) {
        sep = '\t';
        outfile = stdout;
    } else {
        sep = '\n';
        outfile = fopen(filename, "w");
        if (outfile == NULL) {
            printf("Can't open output file.");
            exit(-1);
        }
    }

    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++)
            fprintf(outfile, "%6.2f%c", a[i][j], sep);
        fprintf(outfile, "\n");
    }

    if (outfile != stdout)
        fclose(outfile);
}

static inline void compute_row_range(int start, int end, double *out_diff)
{
    double local = 0.0;

    for (int i = start; i < end; i++) {
        for (int j = 1; j < N - 1; j++) {
            double val = 0.25 * (u[i - 1][j] + u[i + 1][j] +
                                 u[i][j - 1] + u[i][j + 1]);
            w[i][j] = val;
            double d = fabs(val - u[i][j]);
            if (d > local)
                local = d;
        }
    }

    *out_diff = local;
}

static void *worker_func(void *arg)
{
    thread_data_t *td = (thread_data_t *) arg;

    for (;;) {
        compute_row_range(td->start_row, td->end_row, &td->local_diff);

        if (sem_post(&sem_worker_done) == -1) {
            perror("sem_post");
            exit(-1);
        }

        if (sem_wait(&sem_master_go) == -1) {
            perror("sem_wait");
            exit(-1);
        }

        if (terminate)
            break;
    }

    getrusage(RUSAGE_THREAD, &td->usage);
    return NULL;
}

static void init_solver(void)
{
    if (sem_init(&sem_worker_done, 0, 0) == -1) {
        perror("sem_init");
        exit(-1);
    }
    if (sem_init(&sem_master_go, 0, 0) == -1) {
        perror("sem_init");
        exit(-1);
    }

    terminate = 0;
    final_diff = 0.0;

    threads = (pthread_t *) malloc(T * sizeof(pthread_t));
    workers = (thread_data_t *) malloc(T * sizeof(thread_data_t));
    if (!threads || !workers) {
        perror("malloc");
        exit(-1);
    }

    int base = (M - 2) / T;
    int rem = (M - 2) % T;
    int row = 1;

    for (int i = 0; i < T; i++) {
        workers[i].id = i;
        workers[i].start_row = row;
        int n = base + (i < rem ? 1 : 0);
        workers[i].end_row = row + n;
        row = workers[i].end_row;
        workers[i].local_diff = 0.0;
        memset(&workers[i].usage, 0, sizeof(struct rusage));
    }
}

static void cleanup_solver(void)
{
    sem_destroy(&sem_worker_done);
    sem_destroy(&sem_master_go);
    free(threads);
    free(workers);
    threads = NULL;
    workers = NULL;
}

static void print_thread_metrics(const thread_data_t *td)
{
    printf("Thread %d has completed - user: %.4f s, system: %.4f s\n",
           td->id,
           td->usage.ru_utime.tv_sec + td->usage.ru_utime.tv_usec / 1000000.0,
           td->usage.ru_stime.tv_sec + td->usage.ru_stime.tv_usec / 1000000.0);
}

static int find_steady_state(void)
{
    int i, its;

    init_solver();

    for (i = 0; i < T; i++) {
        if (pthread_create(&threads[i], NULL, worker_func,
                           (void *) &workers[i]) != 0) {
            perror("pthread_create");
            exit(-1);
        }
    }

    for (its = 1; its <= max_its; its++) {
        for (i = 0; i < T; i++) {
            if (sem_wait(&sem_worker_done) == -1) {
                perror("sem_wait");
                exit(-1);
            }
        }

        double diff = 0.0;
        for (i = 0; i < T; i++) {
            if (workers[i].local_diff > diff)
                diff = workers[i].local_diff;
        }
        final_diff = diff;

        double **tmp = u;
        u = w;
        w = tmp;

        if (diff <= EPSILON || its == max_its)
            terminate = 1;

        for (i = 0; i < T; i++) {
            if (sem_post(&sem_master_go) == -1) {
                perror("sem_post");
                exit(-1);
            }
        }

        if (terminate)
            break;
    }

    for (i = 0; i < T; i++)
        pthread_join(threads[i], NULL);

    for (i = 0; i < T; i++)
        print_thread_metrics(&workers[i]);

    struct rusage master_usage;
    getrusage(RUSAGE_THREAD, &master_usage);
    printf("find_steady_state - user: %.4f s, system: %.4f s\n",
           master_usage.ru_utime.tv_sec + master_usage.ru_utime.tv_usec / 1000000.0,
           master_usage.ru_stime.tv_sec + master_usage.ru_stime.tv_usec / 1000000.0);

    cleanup_solver();

    return its;
}
