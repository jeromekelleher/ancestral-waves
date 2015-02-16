all: discrete

discrete: discrete.c
	gcc -Wall -g -O2 -o discrete -l gsl -lgslcblas -lconfig discrete.c

numerics: numerics.c
	gcc -Wall -g -o numerics -l gsl -lgslcblas numerics.c
