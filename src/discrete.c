/*
 * Simulations of the ancestral wave in 1D for a discrete deme Wright-Fisher
 * model.
 */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <stdarg.h>
#include <limits.h>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <libconfig.h>

#define MODEL_WF 0
#define MODEL_MORAN 1

/* From version 1.4, libconfig has swapped long for int as the integer type.
 * The LIBCONFIG_VER_MAJOR macro was introduced at version 1.4 also, so this 
 * should be a safe test.
 */
#ifdef LIBCONFIG_VER_MAJOR
typedef int libconfig_int;
#else
typedef long libconfig_int;
#endif

typedef struct {
    unsigned int model;     /* Wright-Fisher or Moran dynamics */
    unsigned int N;         /* deme size */
    unsigned int L;         /* number of demes */
    double m;               /* migration probability */
    unsigned int num_loci;
    long random_seed;
    int verbosity;
    unsigned int max_generations;
    unsigned int output_frequency;
    unsigned int source_deme;
    char* output_prefix;
    /* state */
    gsl_rng *rng;
    unsigned int time;
    unsigned int buffer;
    unsigned int *population[2]; 
    unsigned int *count[2]; 
} sim_t;

void *
xmalloc(size_t size)
{
    register void *value = malloc(size);
    if (value == NULL) {
        perror("virtual memory exhausted");
        abort();
    }
    return value;
}

/*
 * Parses the specified string into a long and assigns the value into 
 * the specified pointer. Returns EINVAL if the string cannot be 
 * converted to double or if min <= x <= max does not hold; returns 0 if 
 * the value is converted successfully.
 */
int 
parse_long(const char *str, long *value, const long min, 
        const long max)
{
    int ret = 0;
    long x;
    char *tail; 
    x = strtol(str, &tail, 10);
    if (tail[0] != '\0') {
        ret = EINVAL; 
    } else if (min > x || max < x) {
        ret = EINVAL;
    } else {
        *value = x;
    }
    return ret;
}

void 
fatal_error(const char *msg, ...)
{
    va_list argp;
    fprintf(stderr, "discrete:");
    va_start(argp, msg);
    vfprintf(stderr, msg, argp);
    va_end(argp);
    fprintf(stderr, "\n");
    exit(EXIT_FAILURE);
}

static void 
sim_alloc(sim_t *self)
{
    unsigned int k;
    size_t size = self->N * self->L * sizeof(unsigned int);
    for (k = 0; k < 2; k++) {
        self->population[k] = xmalloc(size);
        self->count[k] = xmalloc(self->L * sizeof(unsigned int));
    }
    self->rng = gsl_rng_alloc(gsl_rng_default);
    gsl_rng_set(self->rng, self->random_seed);
}

static void
sim_free(sim_t *self)
{
    unsigned int k;
    for (k = 0; k < 2; k++) {
        free(self->population[k]);
        free(self->count[k]);
    }
    free(self->output_prefix);
    gsl_rng_free(self->rng);
}

void 
sim_print_count(sim_t *self, FILE *f)
{
    unsigned int deme;
    for (deme = 0; deme < self->L; deme++) {
        fprintf(f, "%d ", self->count[self->buffer][deme]);
    }
    fprintf(f, "\n");
}

void 
sim_print_state(sim_t *self)
{
    unsigned int deme, ind, k, s;
    unsigned int genetic_material = 0;
    printf("state @ t = %d\n", self->time);
    for (deme = 0; deme < self->L; deme++) {
        printf("%03d: ", deme); 
        s = 0;
        for (ind = 0; ind < self->N; ind++) {
            k = deme * self->N + ind;
            printf("%4d ", self->population[self->buffer][k]);
            s += self->population[self->buffer][k] != 0; 
            genetic_material += self->population[self->buffer][k]; 
        }
        printf("\t%d\n", s); 
        assert(s == self->count[self->buffer][deme]);
    }
    if (self->num_loci != 0) {
        assert(genetic_material == self->num_loci);
    }
}

/*
 * Sets up the initial conditions for the simulation.
 */
static void 
sim_initialise(sim_t *self)
{
    unsigned initial_value = self->num_loci == 0 ? 1 : self->num_loci;
    memset(self->population[0], 0, self->N * self->L * sizeof(unsigned int));
    memset(self->count[0], 0, self->L * sizeof(unsigned int));
    self->population[0][self->source_deme * self->N] = initial_value;
    self->count[0][self->source_deme] = 1;
    self->time = 0;
    self->buffer = 0;
}

/*
 * Implements a single generation in the Wright-Fisher model.
 */
static void 
sim_wf_generation(sim_t *self)
{
    int source_deme, dest_deme, source_ind, dest_ind, offset, source, dest;
    unsigned int *population, *new_population;
    unsigned int *count, *new_count;
    unsigned int parent, v;
    double u;
    population = self->population[self->buffer];
    new_population = self->population[(self->buffer + 1) % 2];
    count = self->count[self->buffer];
    new_count = self->count[(self->buffer + 1) % 2];
    memset(new_population, 0, self->N * self->L * sizeof(unsigned int));
    memset(new_count, 0, self->L * sizeof(unsigned int));
    for (source_deme = 0; source_deme < self->L; source_deme++) {
        if (count[source_deme] != 0) {
            for (source_ind = 0; source_ind < self->N; source_ind++) {
                source = source_deme * self->N + source_ind;
                if (population[source] != 0) {
                    for (parent = 0; parent < 2; parent++) {
                        /* Choose the destination deme */
                        u = gsl_rng_uniform(self->rng);
                        offset = 0;
                        if (u < self->m / 2) {
                            offset = -1;
                        } else if (u < self->m) {
                            offset = 1;
                        }
                        dest_deme = source_deme + offset;
                        /* Any migration outside is ignored */
                        if (dest_deme < 0) {
                            dest_deme = 0;
                        } else if (dest_deme >= self->L) {
                            dest_deme = self->L - 1;
                        }
                        /* Choose the destination individual */
                        dest_ind = gsl_rng_uniform_int(self->rng, self->N);
                        /* Now update the destination individual */
                        dest = dest_deme * self->N + dest_ind;
                        if (self->num_loci == 0) { 
                            /* Pedigree ancestry */
                            new_count[dest_deme] += 
                                    (new_population[dest] ^ population[source]);
                            new_population[dest] |= population[source]; 
                        } else {
                            /* Genetic ancestry */
                            if (parent == 0) {
                                v = gsl_ran_binomial(self->rng, 0.5, population[source]);
                                population[source] -= v;
                            } else {
                                v = population[source];
                            }
                            if (new_population[dest] == 0 && v != 0) {
                                new_count[dest_deme]++;
                            }
                            new_population[dest] += v;
                        }
                    }
                }
            }
        }
    }   
    self->buffer = (self->buffer + 1) % 2;
}

/*
 * Implements a single event in the Moran model. In this model, we choose a deme uniformly
 * at random and a individual uniformly at random from this deme. We then choose two 
 * individual according to the migration rules, and these are assinged any ancestral 
 * material that the individual carries.
 */
static void 
sim_moran_event(sim_t *self)
{
    int source_deme, source_ind, dest_deme, dest_ind, source, dest, offset, 
            parent;
    unsigned int *population = self->population[self->buffer];
    unsigned int *count = self->count[self->buffer];
    unsigned int am, v;
    double u;

    source_deme = gsl_rng_uniform_int(self->rng, self->L);
    source_ind = gsl_rng_uniform_int(self->rng, self->N);
    source = source_deme * self->N + source_ind;
    am = population[source];
    if (am != 0) {
        population[source] = 0;
        count[source_deme]--;
        for (parent = 0; parent < 2; parent++) {
            /* Choose the destination deme */
            u = gsl_rng_uniform(self->rng);
            offset = 0;
            if (u < self->m / 2) {
                offset = -1;
            } else if (u < self->m) {
                offset = 1;
            }
            dest_deme = source_deme + offset;
            /* Any migration outside is ignored */
            if (dest_deme < 0) {
                dest_deme = 0;
            } else if (dest_deme >= self->L) {
                dest_deme = self->L - 1;
            }
            /* Choose the destination individual */
            dest_ind = gsl_rng_uniform_int(self->rng, self->N);
            /* Now update the destination individual */
            dest = dest_deme * self->N + dest_ind;
            //printf("dest = %d\t%d\n", dest_deme, dest_ind);
            if (self->num_loci == 0) { 
                /* Pedigree ancestry */
                count[dest_deme] += (population[dest] ^ am);
                population[dest] |= am;
            } else {
                /* Genetic ancestry */
                if (parent == 0) {
                    v = gsl_ran_binomial(self->rng, 0.5, am);
                    am -= v;
                } else {
                    v = am;
                }
                if (population[dest] == 0 && v != 0) {
                    count[dest_deme]++;
                }
                population[dest] += v;
            }
        }
    }
}


static void
sim_output_count(sim_t *self)
{   
    char filename[8192];
    FILE *f = NULL;
    snprintf(filename, 8192, "%s%d_%ld.dat", self->output_prefix, self->time, 
            self->random_seed);
    f = fopen(filename, "w");
    if (f == NULL) {
        fatal_error("cannot open %s: %s", filename, strerror(errno));
    }
    sim_print_count(self, f);
    if (fclose(f) != 0) {
        fatal_error("cannot close %s: %s", filename, strerror(errno));
    }
}

static void
sim_run(sim_t *self)
{
    int g, j;
    for (g = 0; g <= self->max_generations; g++) {
        if (g % self->output_frequency == 0) {
            sim_output_count(self);
        }
        if (self->verbosity >= 1) {
            sim_print_count(self, stdout);
        }
        if (self->verbosity >= 2) {
            sim_print_state(self); 
        }
        if (self->model == MODEL_WF) {
            sim_wf_generation(self);
        } else {
            for (j = 0; j < self->L * self->N; j++) {
                sim_moran_event(self);
            }
        }
        self->time++;
    }
}

static void
sim_read_config(sim_t *self, const char *filename)
{
    int err;
    libconfig_int tmp;
    const char *str;
    size_t s;
    config_t *config = xmalloc(sizeof(config_t)); 
    config_init(config);
    err = config_read_file(config, filename);
    if (err == CONFIG_FALSE) {
        fatal_error("configuration error:%s at line %d in file %s\n", 
                config_error_text(config), config_error_line(config), 
                filename);
    }
    if (config_lookup_int(config, "verbosity", &tmp) == CONFIG_FALSE) {
        fatal_error("verbosity is a required parameter");
    }
    self->verbosity = tmp;
    if (config_lookup_int(config, "num_demes", &tmp) == CONFIG_FALSE) {
        fatal_error("num_demes is a required parameter");
    }
    self->L= tmp;
    if (config_lookup_int(config, "num_loci", &tmp) == CONFIG_FALSE) {
        fatal_error("num_loci is a required parameter");
    }
    self->num_loci = tmp;
    if (config_lookup_int(config, "deme_size", &tmp) == CONFIG_FALSE) {
        fatal_error("deme_size is a required parameter");
    }
    self->N = tmp;
    if (config_lookup_int(config, "source_deme", &tmp) == CONFIG_FALSE) {
        fatal_error("source_deme is a required parameter");
    }
    self->source_deme = tmp;
    if (config_lookup_int(config, "max_generations", &tmp) == CONFIG_FALSE) {
        fatal_error("max_generations is a required parameter");
    }
    self->max_generations = tmp;
    if (config_lookup_int(config, "output_frequency", &tmp) == CONFIG_FALSE) {
        fatal_error("output_frequency is a required parameter");
    }
    self->output_frequency = tmp;
    if (config_lookup_float(config, "migration_rate", 
            &self->m) == CONFIG_FALSE) {
        fatal_error("migration_rate is a required parameter");
    }
    if (config_lookup_string(config, "model", &str) == CONFIG_FALSE) {
        fatal_error("model is a required parameter");
    }
    if (strcmp(str, "Wright-Fisher") == 0) {
        self->model = MODEL_WF;
    } else if (strcmp(str, "Moran") == 0) {
        self->model = MODEL_MORAN;
    } else {
        fatal_error("model must be 'Wright-Fisher' or 'Moran'");
    }
    if (config_lookup_string(config, "output_prefix", &str) == CONFIG_FALSE) {
        fatal_error("output_prefix is a required parameter");
    }
    s = strlen(str);
    self->output_prefix = xmalloc(s + 1);
    strcpy(self->output_prefix, str);
    config_destroy(config);
    free(config);
}

int 
main(int argc, char** argv)
{
    sim_t *self = xmalloc(sizeof(sim_t));
    if (argc != 3) {
        fatal_error("usage: sim <configfile> <seed>");
    }
    if (parse_long(argv[2], &self->random_seed, 0, LONG_MAX) != 0) {
        fatal_error("cannot parse seed '%s'", argv[1]);
    }   
    sim_read_config(self, argv[1]);
    sim_alloc(self);
    sim_initialise(self);
    sim_run(self);
    sim_free(self);
    free(self);
    return 0;
}
