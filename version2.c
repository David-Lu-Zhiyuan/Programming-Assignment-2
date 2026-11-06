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

typedef struct solver_ctx solver_ctx_t;

typedef struct {
    int id;
    int start_row;
    int end_row;
    double local_diff;
    struct rusage usage;
    solver_ctx_t *ctx;
} thread_data_t;

struct solver_ctx {
    double **u;
    double **w;
    int M, N;
    int T;
    long max_its;
    double final_diff;

    pthread_t *threads;
    thread_data_t *workers;

    sem_t sem_worker_done;
    sem_t sem_master_go;

    volatile int terminate;

    char filename[255];
};

static void allocate_2d_array(int r, int c, double ***a);
static void initialize_array(double **a, int M, int N);
static void print_solution(const char *filename, double **a, int M, int N);
static inline void sem_wait_retry(sem_t *s);
static inline void compute_row_range(solver_ctx_t *ctx, int start, int end,
                                     double *out_diff);
static void *worker_func(void *arg);
static void init_solver(solver_ctx_t *ctx);
static void cleanup_solver(solver_ctx_t *ctx);
static void print_thread_metrics(const thread_data_t *td);
static int find_steady_state(solver_ctx_t *ctx);

int main(int argc, char *argv[])
{
    solver_ctx_t ctx;
    struct timeval stime, etime;
    struct rusage usage;
    double elapsed;
    int its;

    memset(&ctx, 0, sizeof(ctx));
    ctx.M = 200;
    ctx.N = 200;
    ctx.T = 2;
    ctx.max_its = 1000000;

    if (argc == 3) {
        ctx.M = atoi(argv[1]);
        ctx.N = atoi(argv[2]);
    } else if (argc == 4) {
        ctx.M = atoi(argv[1]);
        ctx.N = atoi(argv[2]);
        ctx.T = atoi(argv[3]);
    } else if (argc != 1) {
        printf("Usage: %s [ <rows> <cols> [<threads>] ]\n", argv[0]);
        exit(-1);
    }

    if (ctx.M < 3 || ctx.N < 3 || ctx.T <= 0) {
        printf("Invalid parameters: rows>=3, cols>=3, threads>0 required.\n");
        exit(-1);
    }

    printf("Problem size: M=%d, N=%d\nThread count: T=%d\n",
           ctx.M, ctx.N, ctx.T);

    sprintf(ctx.filename, "jacobi_sema_%d_%d_%d.dat", ctx.M, ctx.N, ctx.T);

    allocate_2d_array(ctx.M, ctx.N, &ctx.u);
    allocate_2d_array(ctx.M, ctx.N, &ctx.w);
    initialize_array(ctx.u, ctx.M, ctx.N);
    initialize_array(ctx.w, ctx.M, ctx.N);

    gettimeofday(&stime, NULL);
    its = find_steady_state(&ctx);
    gettimeofday(&etime, NULL);

    elapsed = ((etime.tv_sec * 1000000 + etime.tv_usec) -
               (stime.tv_sec * 1000000 + stime.tv_usec)) / 1000000.0;

    printf("Converged after %d iterations with error: %8.6f.\n",
           its, ctx.final_diff);
    printf("Elapsed time = %8.4f sec.\n", elapsed);

    getrusage(RUSAGE_SELF, &usage);
    printf("Program completed - user: %.4f s, system: %.4f s\n",
           usage.ru_utime.tv_sec + usage.ru_utime.tv_usec / 1000000.0,
           usage.ru_stime.tv_sec + usage.ru_stime.tv_usec / 1000000.0);
    printf("no. of context switches: vol %ld, invol %ld\n\n",
           usage.ru_nvcsw, usage.ru_nivcsw);

    print_solution(ctx.filename, ctx.u, ctx.M, ctx.N);

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

static inline void sem_wait_retry(sem_t *s)
{
    while (sem_wait(s) == -1) {
        if (errno != EINTR) {
            perror("sem_wait");
            exit(-1);
        }
    }
}

static inline void compute_row_range(solver_ctx_t *ctx, int start, int end,
                                     double *out_diff)
{
    double **u = ctx->u;
    double **w = ctx->w;
    int N = ctx->N;
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
    solver_ctx_t *ctx = td->ctx;

    for (;;) {
        compute_row_range(ctx, td->start_row, td->end_row, &td->local_diff);

        if (sem_post(&ctx->sem_worker_done) == -1) {
            perror("sem_post");
            exit(-1);
        }

        sem_wait_retry(&ctx->sem_master_go);

        if (ctx->terminate)
            break;
    }

    getrusage(RUSAGE_THREAD, &td->usage);
    return NULL;
}

static void init_solver(solver_ctx_t *ctx)
{
    if (sem_init(&ctx->sem_worker_done, 0, 0) == -1) {
        perror("sem_init");
        exit(-1);
    }
    if (sem_init(&ctx->sem_master_go, 0, 0) == -1) {
        perror("sem_init");
        exit(-1);
    }

    ctx->terminate = 0;
    ctx->final_diff = 0.0;

    ctx->threads = (pthread_t *) malloc(ctx->T * sizeof(pthread_t));
    ctx->workers = (thread_data_t *) malloc(ctx->T * sizeof(thread_data_t));
    if (!ctx->threads || !ctx->workers) {
        perror("malloc");
        exit(-1);
    }

    int base = (ctx->M - 2) / ctx->T;
    int rem = (ctx->M - 2) % ctx->T;
    int row = 1;

    for (int i = 0; i < ctx->T; i++) {
        ctx->workers[i].id = i;
        ctx->workers[i].start_row = row;
        int n = base + (i < rem ? 1 : 0);
        ctx->workers[i].end_row = row + n;
        row = ctx->workers[i].end_row;
        ctx->workers[i].local_diff = 0.0;
        ctx->workers[i].ctx = ctx;
        memset(&ctx->workers[i].usage, 0, sizeof(struct rusage));
    }
}

static void cleanup_solver(solver_ctx_t *ctx)
{
    sem_destroy(&ctx->sem_worker_done);
    sem_destroy(&ctx->sem_master_go);
    free(ctx->threads);
    free(ctx->workers);
    ctx->threads = NULL;
    ctx->workers = NULL;
}

static void print_thread_metrics(const thread_data_t *td)
{
    printf("Thread %d has completed - user: %.4f s, system: %.4f s\n",
           td->id,
           td->usage.ru_utime.tv_sec + td->usage.ru_utime.tv_usec / 1000000.0,
           td->usage.ru_stime.tv_sec + td->usage.ru_stime.tv_usec / 1000000.0);
}

static int find_steady_state(solver_ctx_t *ctx)
{
    int i, its;

    init_solver(ctx);

    for (i = 0; i < ctx->T; i++) {
        if (pthread_create(&ctx->threads[i], NULL, worker_func,
                           (void *) &ctx->workers[i]) != 0) {
            perror("pthread_create");
            exit(-1);
        }
    }

    for (its = 1; its <= ctx->max_its; its++) {
        for (i = 0; i < ctx->T; i++)
            sem_wait_retry(&ctx->sem_worker_done);

        double diff = 0.0;
        for (i = 0; i < ctx->T; i++) {
            if (ctx->workers[i].local_diff > diff)
                diff = ctx->workers[i].local_diff;
        }
        ctx->final_diff = diff;

        double **tmp = ctx->u;
        ctx->u = ctx->w;
        ctx->w = tmp;

        if (diff <= EPSILON || its == ctx->max_its)
            ctx->terminate = 1;

        for (i = 0; i < ctx->T; i++) {
            if (sem_post(&ctx->sem_master_go) == -1) {
                perror("sem_post");
                exit(-1);
            }
        }

        if (ctx->terminate)
            break;
    }

    for (i = 0; i < ctx->T; i++)
        pthread_join(ctx->threads[i], NULL);

    for (i = 0; i < ctx->T; i++)
        print_thread_metrics(&ctx->workers[i]);

    struct rusage master_usage;
    getrusage(RUSAGE_THREAD, &master_usage);
    printf("find_steady_state - user: %.4f s, system: %.4f s\n",
           master_usage.ru_utime.tv_sec + master_usage.ru_utime.tv_usec / 1000000.0,
           master_usage.ru_stime.tv_sec + master_usage.ru_stime.tv_usec / 1000000.0);

    cleanup_solver(ctx);

    return its;
}
