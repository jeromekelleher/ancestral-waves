all: discrete

discrete: discrete.c
	gcc -Wall -g -O2 -o discrete -l gsl -lgslcblas -lconfig discrete.c

clean:
	rm -f discrete
