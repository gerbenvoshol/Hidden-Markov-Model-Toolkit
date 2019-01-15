/* hmmtk.h - v1.01 - simple discrete (DHMM) and continuous Hidden Markov Model (CHMM) Toolkit -- GNU GPL
					no warranty is offered or implied; use this code at your own risk

	 This is a single header file with a bunch of useful HMM functions

 ============================================================================
	 You MUST

			#define HMMTK_DEFINE

	 in EXACTLY _one_ C or C++ file that includes this header, BEFORE the
	 include, like this:

			#define HMMTK_DEFINE
			#include "hmmtk.h"

	 All other files should just #include "hmmtk.h" without the #define.
 ============================================================================

 Version History
 		1.01  Fixed memory leak, fixed the chmm_outprob() function (= P(x|mu, var)), all 
 		      values of A, B and pi are now stored as ln() internally instead of converting 
 		      them in the HMM functions
		1.00  Initial release containing basic discrete and continuous hmm functions

 CITATION

 If you use this HMM Toolkit in a publication, please reference:
 Voshol, G.P. (2019). HMMTK: A simple HMM Toolkit (Version 1.0) [Software]. 
 Available from https://github.com/gerbenvoshol/Hidden-Markov-Model-Toolkit

 LICENSE

 This program is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation; either version 2 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

 CREDITS

 The basic framework (including variable names, subroutine names) is highly similar to the
 UMDHMM package (Tapas Kanungo, see references), but the actual implementation of those
 subroutines is different. For example UMDHMM uses scaling to achieve numerical stability, but
 in this package we choose to use the logarithm according to the paper by Mann (2006).

 REFERENCES

 - Rabiner, L. R. and B. H. Juang, "Fundamentals of Speech Recognition,"
   Prentice Hall, 1993.
 - Rabiner, L. R., "A Tutorial on Hidden Markov Models and Selected
   Applications in Speech Recognition, Prov. of IEEE, vol. 77, no. 2,
   pp. 257-286, 1989.
 - Rabiner, L. R., and B. H. Juang, "An Introduction to Hidden Markov Models,"
   IEEE ASSP Magazine, vol. 3, no. 1, pp. 4-16, Jan. 1986.
 - Kanungo, T., UMDHMM: Hidden Markov Model Toolkit, in Extended Finite State Models of Language,
   A. Kornai (editor), Cambridge University Press, 1999.
 - P. Mann, Tobias. (2006). Numerically Stable Hidden Markov Model Implementation.
 - Numerical Recipes in C. The Art of Scientific Computing, 2nd Edition, 1992
 - O'Neill, M. E., PCG: A Family of Simple Fast Space-Efficient Statistically
   Good Algorithms for Random Number Generation", Harvey Mudd College, 2014

  Written by Gerben Voshol.
 */

#ifndef HMMTK__H
#define HMMTK__H

#include <stdlib.h>
#include <sys/types.h>
#include <time.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <float.h>

#ifdef __cplusplus
#define HMM_EXTERN   extern "C"
#else
#define HMM_EXTERN   extern
#endif

#ifndef MAX
#define MAX(x,y)        ((x) > (y) ? (x) : (y))
#endif
#ifndef MIN
#define MIN(x,y)        ((x) < (y) ? (x) : (y))
#endif

/* The accuracy for BaumWelch */
#define DELTA (1E-4)
//#define DELTA (DBL_EPSILON)

#define MIN_COV (1E-4)
//#define MIN_COV DBL_EPSILON

/* --- Section with general hmm functions --- */
/* PCG Random number generator as a replacement for rand() */
/* Get a random seed */
HMM_EXTERN uint64_t hmm_seed(void);
/* Set a seed */
HMM_EXTERN void hmm_srand(uint64_t nseed);
/* Get a random number */
HMM_EXTERN double hmm_rand(void);
/* Get a normally distributed random number */
HMM_EXTERN double hmm_norm_rand(double mu, double sigma);

/* Simple file IO */
/* Reads an entire file into an array of strings, needs only a single call to free */
HMM_EXTERN char **hmm_fgetlns(char *filename, int *number_of_lines);
/* Dynamic allocation version of fgets(), capable of reading unlimited line lengths. */
HMM_EXTERN char *hmm_fgetln(char **buf, int *n, FILE *fp);

/* --- Section with functions for discrete HMMs (DHMM) --- */
typedef struct {
	int N;      /* The number of states for the model */
	char *nN;   /* The names of states */
	int M;      /* The number of distinct observations symbols per state, i.e., the discrete alphabet size */
	char *nM;   /* The names of symbols */
	double **A; /* The NxN state transition probability distribution given in the form of a matrix A
				 * A[1..N][1..N). a[i][j] is the transition prob of going from state i at time t to state j at time t+1 */
	double **B; /* The NxM observation symbol probability distribution given in the form of a matrix B = {bj(k)}:
				 * B[1..N][1..M]. b[j][k] is the probability of observing symbol k in state j */
	double *pi; /* The initial state distribution vector (prior) π = {πi}
				 * pi[1..N) pi[i] is the initial state distribution. */
} DHMM;

/* HMM load/save/init routines */
HMM_EXTERN void dhmm_load(char *filename, DHMM *hmm);
HMM_EXTERN void dhmm_free(DHMM *hmm);
HMM_EXTERN void dhmm_init(DHMM *hmm, int N, char *nN, int M, char *nM, int seed);
HMM_EXTERN void dhmm_copy(DHMM *hmm1, DHMM *hmm2);
HMM_EXTERN void dhmm_save(char *filename, DHMM *hmm);

/* Simple observation sequence saving/loading */
/* convert a C-string (0 indexed) to a sequence of observations (O) containing T observations */
HMM_EXTERN void dhmm_cstr2seq(DHMM *hmm, char *cstr, int *T, int **rO);
/* convert a matrix of C-strings (0 indexed) to sequences of observations (O) containing T observations
 * Note: this is used to train a HMM using multiple sequences using dhmm_baumwelch_multi
 */
HMM_EXTERN int dhmm_cstrmat2seq(DHMM *hmm, char **cstr, int **T, int *L, int ***rO);

/* Generate a sequence of observations (O) and accompanying states (q) with a length (T) and optionally a sees */
HMM_EXTERN void dhmm_genseq(DHMM *hmm, int seed, int T, int *O, int *q);

/* Load a sequence (O) from file (filename) containing T observations */
HMM_EXTERN void dhmm_loadseq(char *filename, int *T, int **rO);
HMM_EXTERN void dhmm_saveseq(char *filename, int T, int *O);

/* The main HMM algorithms */
/* Probability, that the sequence of symbols O1, . . . , OT is generated, and the system is in state N at time t. */
HMM_EXTERN void dhmm_forward(DHMM *hmm, int T, int *O, double **alpha, double *prob);
/* Probability, that the system starts in state N at time t and then generates the sequence of symbols O1,...,OT. */
HMM_EXTERN void dhmm_backward(DHMM *hmm, int T, int *O, double **beta, double *prob);
/* Probability, with which the most probable state path generates the sequence of symbols (O1,O2,...,OT) and the system is in state N at time t. */
HMM_EXTERN void dhmm_viterbi(DHMM *hmm, int T, int *O, double **delta, int **psi, int *path, double *prob);
/* Probability maximized HMM (Given an unknown HMM and a sequence of observations, find parameters that maximize P(O | M)*/
HMM_EXTERN void dhmm_baumwelch(DHMM *hmm, int T, int *O, double **alpha, double **beta, double **gamma, int *pniter, double *plogprobinit, double *plogprobfinal);
/* Probability maximized HMM (Given an unknown HMM and several sequences of observations, find parameters that maximize P(OL | M)*/
HMM_EXTERN void dhmm_baumwelch_multi(DHMM *hmm, int *T, int L, int **O, double **alpha, double **beta, double **gamma, int *pniter, double *plogprobinit, double *plogprobfinal);
/* Probability maximized HMM (Given an unknown HMM and several sequences of observations, find parameters that maximize P(OL | M)
 * with Tied Observations */
HMM_EXTERN void dhmm_baumwelch_multiwt(DHMM *hmm, int *tied_obs, int *T, int L, int **O, double **alpha, double **beta, double **gamma, int *pniter, double *plogprobinit, double *plogprobfinal);
/* Probability maximized HMM (Given an unknown HMM and several sequences of observations, find parameters that maximize P(OL | M)
 * with Tied Observations and Constrained */
HMM_EXTERN void dhmm_baumwelch_multi_constrained(DHMM *hmm, int *tied_obs, int *T, int L, int **O, char **cconstrmat, double **alpha, double **beta, double **gamma, int *pniter, double *plogprobinit, double *plogprobfinal);

/*
 * Simple doubly linked list implementation.
 *
 * Some of the internal functions ("__xxx") are useful when
 * manipulating whole lists rather than single entries, as
 * sometimes we already know the next/prev entries and we can
 * generate better code by using them directly rather than
 * using the generic single-entry routines.
 */

struct list_head {
	struct list_head *next, *prev;
};

//#define LIST_HEAD_INIT(name) { &(name), &(name) }
#define LIST_HEAD_INIT(name) \
	{ name.next = &(name); name.prev = &(name); }

#define LIST_HEAD(name) \
	struct list_head name = LIST_HEAD_INIT(name)

#define INIT_LIST_HEAD(ptr) do { \
	(ptr)->next = (ptr); (ptr)->prev = (ptr); \
} while (0)

/*
 * Insert a new entry between two known consecutive entries.
 *
 * This is only for internal list manipulation where we know
 * the prev/next entries already!
 */
static __inline__ void __list_add(struct list_head * new_,
                                  struct list_head * prev,
                                  struct list_head * next)
{
	next->prev = new_;
	new_->next = next;
	new_->prev = prev;
	prev->next = new_;
}

/**
 * list_add - add a new entry
 * @new: new entry to be added
 * @head: list head to add it after
 *
 * Insert a new entry after the specified head.
 * This is good for implementing stacks.
 */
static __inline__ void list_add(struct list_head *new_, struct list_head *head)
{
	__list_add(new_, head, head->next);
}

/**
 * list_add_tail - add a new entry
 * @new: new entry to be added
 * @head: list head to add it before
 *
 * Insert a new entry before the specified head.
 * This is useful for implementing queues.
 */
static __inline__ void list_add_tail(struct list_head *new_, struct list_head *head)
{
	__list_add(new_, head->prev, head);
}

/*
 * Delete a list entry by making the prev/next entries
 * point to each other.
 *
 * This is only for internal list manipulation where we know
 * the prev/next entries already!
 */
static __inline__ void __list_del(struct list_head * prev,
                                  struct list_head * next)
{
	next->prev = prev;
	prev->next = next;
}

/**
 * list_del - deletes entry from list.
 * @entry: the element to delete from the list.
 * Note: list_empty on entry does not return true after this, the entry is in an undefined state.
 */
static __inline__ void list_del(struct list_head *entry)
{
	__list_del(entry->prev, entry->next);
}

/**
 * list_del_init - deletes entry from list and reinitialize it.
 * @entry: the element to delete from the list.
 */
static __inline__ void list_del_init(struct list_head *entry)
{
	__list_del(entry->prev, entry->next);
	INIT_LIST_HEAD(entry);
}

/**
 * list_empty - tests whether a list is empty
 * @head: the list to test.
 */
static __inline__ int list_empty(struct list_head *head)
{
	return head->next == head;
}

/**
 * list_splice - join two lists
 * @list: the new list to add.
 * @head: the place to add it in the first list.
 */
static __inline__ void list_splice(struct list_head *list, struct list_head *head)
{
	struct list_head *first = list->next;

	if (first != list) {
		struct list_head *last = list->prev;
		struct list_head *at = head->next;

		first->prev = head;
		head->next = first;

		last->next = at;
		at->prev = last;
	}
}

/**
 * list_entry - get the struct for this entry
 * @ptr:	the &struct list_head pointer.
 * @type:	the type of the struct this is embedded in.
 * @member:	the name of the list_struct within the struct.
 */
#define list_entry(ptr, type, member) \
	((type *)((char *)(ptr)-(unsigned long)(&((type *)0)->member)))

/**
 * list_for_each	-	iterate over a list
 * @pos:	the &struct list_head to use as a loop counter.
 * @head:	the head for your list.
 */
#define list_for_each(pos, head) \
	for (pos = (head)->next; pos != (head); pos = pos->next)

/* --- Section with functions for continuous HMMs (CHMM) --- */
typedef struct {
	double **miu;     /* M means of mixture gaussian*/
	double **cov;     /* M covariance of mixture gaussian*/
	double **cov_inv;
} output;

typedef struct {
	int N;      /* The number of states for the model */
	char *nN;   /* The names of states */
	int D;      /* The dimension of the observation vector */
	char *nD;   /* The names of the individual observations */
	double **A;	/* The NxN state transition probability distribution given in the form of a matrix A
				 * A[1..N][1..N). a[i][j] is the transition prob of going from state i at time t to state j at time t+1 */
	output B;   /* The Gaussian observations with their means and covariance */
	double *pi; /* The initial state distribution vector (prior) π = {πi}
				 * pi[1..N) pi[i] is the initial state distribution. */
} CHMM;

/* TODO: make the features dynamic */
/* NOTE: Do NOT forget to set DIM */
#define DIM 1

typedef struct feature_node {
	double feature[DIM];
	struct list_head list;
} feature_node;

typedef struct sample_node {
	struct feature_node feature_head;
	struct list_head list;
} sample_node;

struct samples {
	struct sample_node * sample_head;
	int sample_count;
	int feature_count_max;
	int *feature_count_per_sample;
	double ***data;
};

/* HMM load/save/init routines */
HMM_EXTERN void chmm_load(char *filename, CHMM *hmm);
HMM_EXTERN void chmm_free(CHMM *hmm);
HMM_EXTERN void chmm_init(CHMM *hmm, int N, char *nN, int D, char *nD, int seed);
HMM_EXTERN void chmm_copy(CHMM *hmm1, CHMM *hmm2);
HMM_EXTERN void chmm_save(char *filename, CHMM *hmm);

/* Simple observation sequence saving/loading */
HMM_EXTERN struct samples *chmm_load_samples(char *filename);
HMM_EXTERN void chmm_rebuild_samples(struct samples *samples);
HMM_EXTERN void chmm_print_samples(struct samples *samples);

/* Generate a sequence of observations (O) and accompanying states (q) with a length (T) and optionally a seed */
HMM_EXTERN void chmm_genseq(CHMM *hmm, int seed, int T, double *O, int *q);

/* The main HMM algorithms */
HMM_EXTERN void chmm_outprob(CHMM *hmm, double **sample, int T, double **outprob);
/* Probability, that the sequence of symbols O1, . . . , OT is generated, and the system is in state N at time t. */
HMM_EXTERN void chmm_forward(CHMM *hmm, int T, double **outprob, double **alpha, double *prob);
/* Probability, that the system starts in state N at time t and then generates the sequence of symbols O1,...,OT. */
HMM_EXTERN void chmm_backward(CHMM *hmm, int T, double **outprob, double **beta, double *prob);
/* Probability, with which the most probable state path generates the sequence of symbols (O1,O2,...,OT) and the system is in state N at time t. */
HMM_EXTERN void chmm_viterbi(CHMM *hmm, int T, double **outprob, double **delta, int **psi, int *path, double *prob);
/* Probability maximized HMM (Given an unknown HMM and several sequences of observations, find parameters that maximize P(OL | M)*/
HMM_EXTERN void chmm_baumwelch(CHMM *hmm, struct samples *p_samples, int *piter, double *plogprobinit, double *plogprobfinal, int maxiter);

/* Public domain functions from Numerical Recipies */
HMM_EXTERN void nrerror();
HMM_EXTERN float *vector(int, int);
HMM_EXTERN float **matrix(int, int, int, int);
HMM_EXTERN float **convert_matrix(float*, int, int, int, int);
HMM_EXTERN double *dvector(int, int);
HMM_EXTERN double **dmatrix(int nrl, int nrh, int ncl, int nch);
HMM_EXTERN double ***d3tensor(long nrl, long nrh, long ncl, long nch, long ndl, long ndh);
HMM_EXTERN char *cvector(int, int);
HMM_EXTERN int *ivector(int, int);
HMM_EXTERN int **imatrix(int, int, int, int);
HMM_EXTERN float **submatrix(float **, int, int, int, int, int, int);
HMM_EXTERN void clear_dvector(double*, int, int);
HMM_EXTERN void clear_dvector_nan(double*, int, int);
HMM_EXTERN void clear_ivector(int*, int, int);
HMM_EXTERN void clear_dmatrix(double**, int, int, int, int);
HMM_EXTERN void clear_dmatrix_nan(double**, int, int, int, int);
HMM_EXTERN void clear_imatrix(int**, int, int, int, int);
HMM_EXTERN void clear_dmatrix3d(double***, int, int, int, int, int, int);
HMM_EXTERN void free_vector(float*, int, int);
HMM_EXTERN void free_dvector(double*, int, int);
HMM_EXTERN void free_cvector(char*, int, int);
HMM_EXTERN void free_ivector(int*, int, int);
HMM_EXTERN void free_matrix(float**, int, int, int, int);
HMM_EXTERN void free_dmatrix(double**, int, int, int, int);
HMM_EXTERN void free_imatrix(int**, int, int, int, int);
HMM_EXTERN void free_submatrix(float**, int, int, int, int);
HMM_EXTERN void free_convert_matrix(float **, int, int, int, int);
HMM_EXTERN void free_d3tensor(double ***t, long nrl, long nrh, long ncl, long nch, long ndl, long ndh);

#ifdef HMMTK_DEFINE

/* Extended exponential function */
double eexp(double x)
{
	if (isnan(x)) {
		return 0.0;
	} else {
		return exp(x);
	}
}

/* Extended logarithm function */
double eln(double x)
{
	if (fabs(x) < DBL_EPSILON) {
		return NAN;
	} else if (x > 0) {
		return log(x);
	} else {
		fprintf(stderr, "Negative input: %lf\n", x);
		return x;
	}
}

/* Extended logarithm sum function */
double elnsum(double x, double y)
{
	if (isnan(x) || isnan(y)) {
		if (isnan(x)) {
			return y;
		} else {
			return x;
		}
	} else if (x > y) {
		return x + eln(1 + exp(y - x));
	} else {
		return y + eln(1 + exp(x - y));
	}
}

/* extended logarithm product function */
double elnproduct(double x, double y)
{
	if (isnan(x) || isnan(y)) {
		return NAN;
	} else {
		return (x + y);
	}
}

/* seed for the random number generator */
static uint64_t seed = 0;

/* Load a HMM from a file pointed to by filename */
void dhmm_load(char *filename, DHMM *hmm)
{
	int i, j, k;
	FILE *fp = NULL;
	char *line = NULL;
	int len = 0;

	fp = fopen(filename, "r");
	if (fp == NULL) {
		fprintf(stderr, "Error: File %s not found\n", filename);
		exit(1);
	}

	line = hmm_fgetln(&line, &len, fp);
	if (strncmp("HMM type: Discrete", line, 18)) {
		fprintf(stderr, "Unsupported HMM format: %s\n", line);
		fprintf(stderr, "Only Discrete or Continuous supported!\n");
		exit(1);
	}
	free(line);

	fscanf(fp, "Number of states (N): %d\n", &(hmm->N));
	fscanf(fp, "Names of states:\n");
	hmm->nN = (char *) cvector(1, hmm->N);
	for (i = 1; i <= hmm->N; i++) {
		fscanf(fp, "%c ", &(hmm->nN[i]));
	}
	fscanf(fp, "\n");

	fscanf(fp, "Number of symbols (M): %d\n", &(hmm->M));
	fscanf(fp, "Names of symbols:\n");
	hmm->nM = (char *) cvector(1, hmm->M);
	for (i = 1; i <= hmm->M; i++) {
		fscanf(fp, "%c ", &(hmm->nM[i]));
	}
	fscanf(fp, "\n");

	fscanf(fp, "Transition matrix (A):\n");
	hmm->A = (double **) dmatrix(1, hmm->N, 1, hmm->N);
	for (i = 1; i <= hmm->N; i++) {
		for (j = 1; j <= hmm->N; j++) {
			fscanf(fp, "%lf", &(hmm->A[i][j]));
			hmm->A[i][j] = eln(hmm->A[i][j]);
		}
		fscanf(fp, "\n");
	}

	fscanf(fp, "Emission probabilities (B):\n");
	hmm->B = (double **) dmatrix(1, hmm->N, 1, hmm->M);
	for (j = 1; j <= hmm->N; j++) {
		for (k = 1; k <= hmm->M; k++) {
			fscanf(fp, "%lf", &(hmm->B[j][k]));
			hmm->B[j][k] = eln(hmm->B[j][k]);
		}
		fscanf(fp, "\n");
	}

	fscanf(fp, "Initial state probability (pi):\n");
	hmm->pi = (double *) dvector(1, hmm->N);
	for (i = 1; i <= hmm->N; i++) {
		fscanf(fp, "%lf", &(hmm->pi[i]));
		hmm->pi[i] = eln(hmm->pi[i]);
	}

	fclose(fp);
}

void dhmm_free(DHMM *hmm)
{
	free_cvector(hmm->nN, 1, hmm->N);
	free_cvector(hmm->nM, 1, hmm->M);
	free_dmatrix(hmm->A, 1, hmm->N, 1, hmm->N);
	free_dmatrix(hmm->B, 1, hmm->N, 1, hmm->M);
	free_dvector(hmm->pi, 1, hmm->N);
}

/* Initialize a HMM (A, B and pi) with random values */
void dhmm_init(DHMM *hmm, int N, char *nN, int M, char *nM, int seed)
{
	int i, j, k;
	double sum;

	/* initialize random number generator */
	if (seed) {
		hmm_srand(seed);
	} else {
		hmm_srand(hmm_seed());
	}

	hmm->N = N;
	hmm->nN = (char *) cvector(1, hmm->N);
	for (i = 1; i <= hmm->N; i++) {
		hmm->nN[i] = nN[i];
	}

	hmm->M = M;
	hmm->nM = (char *) cvector(1, hmm->M);
	for (i = 1; i <= hmm->M; i++) {
		hmm->nM[i] = nM[i];
	}

	hmm->A = (double **) dmatrix(1, hmm->N, 1, hmm->N);
	for (i = 1; i <= hmm->N; i++) {
		sum = 0.0;

		for (j = 1; j <= hmm->N; j++) {
			hmm->A[i][j] = hmm_rand();
			sum += hmm->A[i][j];
		}

		for (j = 1; j <= hmm->N; j++) {
			hmm->A[i][j] /= sum;
		}
	}

	hmm->B = (double **) dmatrix(1, hmm->N, 1, hmm->M);
	for (j = 1; j <= hmm->N; j++) {
		sum = 0.0;
		for (k = 1; k <= hmm->M; k++) {
			hmm->B[j][k] = hmm_rand();
			sum += hmm->B[j][k];
		}

		for (k = 1; k <= hmm->M; k++) {
			hmm->B[j][k] /= sum;
		}
	}

	hmm->pi = (double *) dvector(1, hmm->N);
	sum = 0.0;
	for (i = 1; i <= hmm->N; i++) {
		hmm->pi[i] = hmm_rand();
		sum += hmm->pi[i];
	}

	for (i = 1; i <= hmm->N; i++) {
		hmm->pi[i] /= sum;
	}
}

void dhmm_copy(DHMM *hmm1, DHMM *hmm2)
{
	int i, j, k;

	hmm2->N = hmm1->N;
	hmm2->nN = (char *) cvector(1, hmm2->N);
	for (i = 1; i <= hmm2->N; i++) {
		hmm2->nN[i] = hmm1->nN[i];
	}

	hmm2->M = hmm1->M;
	hmm2->nM = (char *) cvector(1, hmm2->M);
	for (i = 1; i <= hmm2->M; i++) {
		hmm2->nM[i] = hmm1->nM[i];
	}

	hmm2->A = (double **) dmatrix(1, hmm2->N, 1, hmm2->N);

	for (i = 1; i <= hmm2->N; i++) {
		for (j = 1; j <= hmm2->N; j++) {
			hmm2->A[i][j] = hmm1->A[i][j];
		}
	}

	hmm2->B = (double **) dmatrix(1, hmm2->N, 1, hmm2->M);
	for (j = 1; j <= hmm2->N; j++) {
		for (k = 1; k <= hmm2->M; k++) {
			hmm2->B[j][k] = hmm1->B[j][k];
		}
	}

	hmm2->pi = (double *) dvector(1, hmm2->N);
	for (i = 1; i <= hmm2->N; i++) {
		hmm2->pi[i] = hmm1->pi[i];
	}
}

#ifdef DBL_DECIMAL_DIG
#define OP_DBL_Digs (DBL_DECIMAL_DIG)
#else
#ifdef DECIMAL_DIG
#define OP_DBL_Digs (DECIMAL_DIG)
#else
#define OP_DBL_Digs (DBL_DIG + 3)
#endif
#endif

void dhmm_save(char *filename, DHMM *hmm)
{
	int i, j, k;
	FILE *fp = NULL;
	int f = 1;

	if (!strcmp(filename, "stdout")) {
		fp = stdout;
		f = 0;
	} else {
		fp = fopen(filename, "w");
	}
	if (fp == NULL) {
		fprintf(stderr, "Error: File %s not found\n", filename);
		exit(1);
	}

	fprintf(fp, "HMM type: Discrete\n");

	fprintf(fp, "Number of states (N): %d\n", hmm->N);
	fprintf(fp, "Names of states:\n");
	for (i = 1; i <= hmm->N; i++) {
		fprintf(fp, "%c ", hmm->nN[i]);
	}
	fprintf(fp, "\n");

	fprintf(fp, "Number of symbols (M): %d\n", hmm->M);
	fprintf(fp, "Names of symbols:\n");
	for (i = 1; i <= hmm->M; i++) {
		fprintf(fp, "%c ", hmm->nM[i]);
	}
	fprintf(fp, "\n");

	fprintf(fp, "Transition matrix (A):\n");
	for (i = 1; i <= hmm->N; i++) {
		for (j = 1; j <= hmm->N; j++) {
			fprintf(fp, "%.*e ", OP_DBL_Digs - 1, eexp(hmm->A[i][j]));
		}
		fprintf(fp, "\n");
	}

	fprintf(fp, "Emission probabilities (B):\n");
	for (j = 1; j <= hmm->N; j++) {
		for (k = 1; k <= hmm->M; k++) {
			fprintf(fp, "%.*e ", OP_DBL_Digs - 1, eexp(hmm->B[j][k]));
		}
		fprintf(fp, "\n");
	}

	fprintf(fp, "Initial state probability (pi):\n");
	for (i = 1; i <= hmm->N; i++) {
		fprintf(fp, "%.*e ", OP_DBL_Digs - 1, eexp(hmm->pi[i]));
	}
	fprintf(fp, "\n\n");

	if (f) {
		fclose(fp);
	}
}

int dhmm_lookup(DHMM *hmm, char c)
{
	int i;

	for (i = 1; i <= hmm->M; i++) {
		if (c == hmm->nM[i]) {
			return i;
		}
	}

	fprintf(stderr, "Could not find \"%c\" in HMM symbols\n", c);
	return 0;
}

void dhmm_cstr2seq(DHMM *hmm, char *cstr, int *T, int **rO)
{
	int *O;
	int i;

	*T = strlen(cstr);

	O = ivector(1, *T);
	for (i = 1; i <= *T; i++) {
		O[i] = dhmm_lookup(hmm, cstr[i - 1]);
	}

	*rO = O;
}

int dhmm_cstrmat2seq(DHMM *hmm, char **cstr, int **rT, int *L, int ***rO)
{
	int **O;
	int i, n;
	int maxT = 0;
	int *T;

	T = ivector(1, *L);
	for (i = 1; i <= *L; i++) {
		T[i] = strlen(cstr[i - 1]);
		if (T[i] > maxT) {
			maxT = T[i];
		}
	}
	*rT = T;

	if (*L == 0) {
		fprintf(stderr, "dhmm_cstrmat2seq: Please indicate a valid number of observations (L).\n");
		exit(1);
	}

	O = imatrix(1, *L, 1, maxT);
	for (i = 1; i <= *L; i++) {
		for (n = 1; n <= T[i]; n++) {
			O[i][n] = dhmm_lookup(hmm, cstr[i - 1][n - 1]);
		}
	}

	*rO = O;

	return maxT;
}

void dhmm_loadseq(char *filename, int *T, int **rO)
{
	int *O;
	int i;
	FILE *fp = NULL;

	fp = fopen(filename, "r");
	if (fp == NULL) {
		fprintf(stderr, "Error: File %s not found\n", filename);
		exit(1);
	}

	fscanf(fp, "T: %d\n", T);
	O = ivector(1, *T);
	for (i = 1; i <= *T; i++) {
		fscanf(fp, "%d", &O[i]);
	}

	*rO = O;

	fclose(fp);
}

void dhmm_saveseq(char *filename, int T, int *O)
{
	int i;
	FILE *fp = NULL;
	int f = 1;

	if (!strcmp(filename, "stdout")) {
		fp = stdout;
		f = 0;
	} else {
		fp = fopen(filename, "w");
	}
	if (fp == NULL) {
		fprintf(stderr, "Error: File %s not found\n", filename);
		exit(1);
	}

	fprintf(fp, "T: %d\n", T);
	for (i = 1; i <= T; i++) {
		fprintf(fp, "%d ", O[i]);
	}
	printf("\n");

	if (f) {
		fclose(fp);
	}
}

int hmm_lookup(DHMM *hmm, char c)
{
	int i;

	for (i = 1; i <= hmm->M; i++) {
		if (c == hmm->nM[i]) {
			return i;
		}
	}

	fprintf(stderr, "Could not find \"%c\" in HMM symbols\n", c);
	return 0;
}

int dhmm_geninitialstate(DHMM *hmm);
int dhmm_gennextstate(DHMM *hmm, int q_t);
int dhmm_gensymbol(DHMM *hmm, int q_t);

void dhmm_genseq(DHMM *hmm, int seed, int T, int *O, int *q)
{
	int t = 1;

	hmm_srand(seed);

	q[1] = dhmm_geninitialstate(hmm);
	O[1] = dhmm_gensymbol(hmm, q[1]);

	for (t = 2; t <= T; t++) {
		q[t] = dhmm_gennextstate(hmm, q[t - 1]);
		O[t] = dhmm_gensymbol(hmm, q[t]);
	}
}

int dhmm_geninitialstate(DHMM *hmm)
{
	double value;
	double cummulative = 0.0;
	int i, q_t;

	value = hmm_rand();
	q_t = hmm->N;
	for (i = 1; i <= hmm->N; i++) {
		cummulative += eexp(hmm->pi[i]);
		if (value < cummulative) {
			q_t = i;
			break;
		}
	}

	return q_t;
}

int dhmm_gennextstate(DHMM *hmm, int q_t)
{
	double value;
	double cummulative = 0.0;
	int j, q_next;

	value = hmm_rand();
	q_next = hmm->N;
	for (j = 1; j <= hmm->N; j++) {
		cummulative += eexp(hmm->A[q_t][j]);
		if (value < cummulative) {
			q_next = j;
			break;
		}
	}

	return q_next;
}

int dhmm_gensymbol(DHMM *hmm, int q_t)
{
	double value;
	double cummulative = 0.0;
	int j, o_t;

	value = hmm_rand();
	o_t = hmm->M;
	for (j = 1; j <= hmm->M; j++) {
		cummulative += eexp(hmm->B[q_t][j]);
		if (value < cummulative) {
			o_t = j;
			break;
		}
	}

	return o_t;
}

void dhmm_forward(DHMM *hmm, int T, int *O, double **alpha, double *prob)
{
	int	i, j; /* state indices */
	int	t;    /* time index */
	double logalpha;

	/* Initialization */
	for (i = 1; i <= hmm->N; i++) {
		alpha[1][i] = elnproduct(hmm->pi[i], hmm->B[i][O[1]]);
	}

	/* Induction */
	for (t = 2; t <= T; t++) {
		for (j = 1; j <= hmm->N; j++) {
			logalpha = NAN;
			for (i = 1; i <= hmm->N; i++) {
				logalpha = elnsum(logalpha, elnproduct(alpha[t - 1][i], hmm->A[i][j]));
			}
			alpha[t][j] = elnproduct(logalpha, hmm->B[j][O[t]]);
		}
	}

	/* Termination */
	*prob = NAN;
	for (i = 1; i <= hmm->N; i++) {
		*prob = elnsum(*prob, alpha[T][i]);
	}
}

void dhmm_backward(DHMM *hmm, int T, int *O, double **beta, double *prob)
{
	int i, j;   /* state indices */
	int t;      /* time index */
	double logbeta;


	/* Initialization */
	for (i = 1; i <= hmm->N; i++) {
		beta[T][i] = 0.0;
	}

	/* Induction */
	for (t = T - 1; t >= 1; t--) {
		for (i = 1; i <= hmm->N; i++) {
			logbeta = NAN;
			for (j = 1; j <= hmm->N; j++) {
				logbeta = elnsum(logbeta, elnproduct(hmm->A[i][j], elnproduct(hmm->B[j][O[t + 1]], beta[t + 1][j])));
			}
			beta[t][i] = logbeta;
		}
	}

	/* Termination */
	*prob = NAN;
	for (i = 1; i <= hmm->N; i++) {
		*prob = elnsum(*prob, elnproduct(elnproduct(beta[1][i], hmm->pi[i]), hmm->B[i][O[1]]));
	}
}

void dhmm_viterbi(DHMM *hmm, int T, int *O, double **delta, int **psi, int *path, double *prob)
{
	int i, j; /* state indices */
	int t;    /* time index */

	int maxvalind;
	double maxval, val;

	/* Initialization  */
	for (i = 1; i <= hmm->N; i++) {
		delta[1][i] = elnproduct(hmm->pi[i], hmm->B[i][O[1]]);
		psi[1][i] = 0;
	}

	/* Recursion */
	for (t = 2; t <= T; t++) {
		for (j = 1; j <= hmm->N; j++) {
			maxval = -DBL_MAX;
			maxvalind = 1;
			for (i = 1; i <= hmm->N; i++) {
				val = elnproduct(delta[t - 1][i], hmm->A[i][j]);
				if (val > maxval) {
					maxval = val;
					maxvalind = i;
				}
			}
			delta[t][j] = elnproduct(maxval, hmm->B[j][O[t]]);
			psi[t][j] = maxvalind;
		}
	}

	/* Termination */
	*prob = -DBL_MAX;
	path[T] = 1;
	for (i = 1; i <= hmm->N; i++) {
		if (delta[T][i] > *prob) {
			*prob = delta[T][i];
			path[T] = i;
		}
	}

	/* Path (state sequence) backtracking */
	for (t = T - 1; t >= 1; t--) {
		path[t] = psi[t + 1][path[t + 1]];
	}
}

void dhmm_compgamma(DHMM *hmm, int T, double **alpha, double **beta, double **gamma)
{
	int i; /* state indices */
	int	t; /* time index */
	double normalizer;

	for (t = 1; t <= T; t++) {
		normalizer = NAN;
		for (i = 1; i <= hmm->N; i++) {
			gamma[t][i] = elnproduct(alpha[t][i], beta[t][i]);
			normalizer = elnsum(normalizer, gamma[t][i]);
		}
		for (i = 1; i <= hmm->N; i++) {
			gamma[t][i] = elnproduct(gamma[t][i], -normalizer);
		}
	}
}

void dhmm_compxi(DHMM *hmm, int T, int *O, double **alpha, double **beta, double ***xi)
{
	int i, j;
	int t;
	double normalizer;

	for (t = 1; t <= T - 1; t++) {
		normalizer = NAN;
		for (i = 1; i <= hmm->N; i++) {
			for (j = 1; j <= hmm->N; j++) {
				xi[t][i][j] = elnproduct(alpha[t][i], elnproduct(hmm->A[i][j], elnproduct(hmm->B[j][O[t + 1]], beta[t + 1][j])));
				normalizer = elnsum(normalizer, xi[t][i][j]);
			}
		}
		for (i = 1; i <= hmm->N; i++) {
			for (j = 1; j <= hmm->N; j++) {
				xi[t][i][j] = elnproduct(xi[t][i][j], -normalizer);
			}
		}
	}
}

void dhmm_baumwelch(DHMM *hmm, int T, int *O, double **alpha, double **beta, double **gamma, int *pniter, double *plogprobinit, double *plogprobfinal)
{
	int	i, j, k;
	int	t, iter = 0;

	double	logprobf, logprobb;

	double ***xi;
	double delta, logprobprev;

	double numerator, denominator;

	dhmm_forward(hmm, T, O, alpha, &logprobf);
	*plogprobinit = logprobf; /* log P(O |intial model) */

	dhmm_backward(hmm, T, O, beta, &logprobb);

	dhmm_compgamma(hmm, T, alpha, beta, gamma);

	xi = d3tensor(1, T, 1, hmm->N, 1, hmm->N);
	dhmm_compxi(hmm, T, O, alpha, beta, xi);

	logprobprev = logprobf;

	do {
		/* reestimate frequency of state i in time t=1 */
		for (i = 1; i <= hmm->N; i++) {
			hmm->pi[i] = eexp(gamma[1][i]);
		}

		/* reestimate transition matrix  and symbol prob in each state */
		for (i = 1; i <= hmm->N; i++) {
			for (j = 1; j <= hmm->N; j++) {
				numerator = NAN;
				denominator = NAN;
				for (t = 1; t <= T - 1; t++) {
					numerator = elnsum(numerator, xi[t][i][j]);
					denominator = elnsum(denominator, gamma[t][i]);
				}
				hmm->A[i][j] = elnproduct(numerator, -denominator);
			}

			for (k = 1; k <= hmm->M; k++) {
				numerator = NAN;
				denominator = NAN;
				for (t = 1; t <= T; t++) {
					if (O[t] == k) {
						numerator = elnsum(numerator, gamma[t][i]);
					}
					denominator = elnsum(denominator, gamma[t][i]);
				}
				hmm->B[i][k] = elnproduct(numerator, -denominator);
			}
		}

		dhmm_forward(hmm, T, O, alpha, &logprobf);
		dhmm_backward(hmm, T, O, beta, &logprobb);

		dhmm_compgamma(hmm, T, alpha, beta, gamma);
		dhmm_compxi(hmm, T, O, alpha, beta, xi);

		/* compute difference between log probability of two iterations */
		delta = logprobf - logprobprev;
		logprobprev = logprobf;
		iter++;

		printf("iteration %i: delta: %lf (%lf)\n", iter, delta, DELTA);
	} while ((delta > DELTA) && (iter < 100000)); /* if log probability does not change much, exit */

	*pniter = iter;
	*plogprobfinal = logprobf; /* log P(O|estimated model) */
	free_d3tensor(xi, 1, T, 1, hmm->N, 1, hmm->N);
}

void dhmm_baumwelch_multi(DHMM *hmm, int *T, int L, int **O, double **alpha, double **beta, double **gamma, int *pniter, double *plogprobinit, double *plogprobfinal)
{
	int	i, j, k;
	int	t, iter = 1, l = 0;

	double	logprobf = NAN, logprobb;
	double prodlogprob = 0.0;

	double ***xi;
	double delta, logprobprev;

	double numerator;
	double denominator;
	double **numeratorA = dmatrix(1, hmm->N, 1, hmm->N);
	double **denominatorA = dmatrix(1, hmm->N, 1, hmm->N);
	double **numeratorB = dmatrix(1, hmm->N, 1, hmm->M);
	double **denominatorB = dmatrix(1, hmm->N, 1, hmm->M);

	double *logpi = dvector(1, hmm->N);
	double pisum = NAN;
	/* START first iteration */
	/* Initialize the numerators and denominators over l */
	pisum = NAN;
	for (i = 1; i <= hmm->N; i++) {
		logpi[i] = NAN;
		for (j = 1; j <= hmm->N; j++) {
			numeratorA[i][j] = NAN;
			denominatorA[i][j] = NAN;
		}
		for (k = 1; k <= hmm->M; k++) {
			numeratorB[i][k] = NAN;
			denominatorB[i][k] = NAN;
		}
	}

	for (l = 1; l <= L; l++) {
		dhmm_forward(hmm, T[l], O[l], alpha, &logprobf);
		dhmm_backward(hmm, T[l], O[l], beta, &logprobb);
		dhmm_compgamma(hmm, T[l], alpha, beta, gamma);

		xi = d3tensor(1, T[l], 1, hmm->N, 1, hmm->N);
		dhmm_compxi(hmm, T[l], O[l], alpha, beta, xi);

		for (i = 1; i <= hmm->N; i++) {
			logpi[i] = elnsum(logpi[i], gamma[1][i]);
			for (j = 1; j <= hmm->N; j++) {
				numerator = NAN;
				denominator = NAN;
				for (t = 1; t <= T[l] - 1; t++) {
					numerator = elnsum(numerator, xi[t][i][j]);
					denominator = elnsum(denominator, gamma[t][i]);
				}
				numeratorA[i][j] = elnsum(numeratorA[i][j], numerator);
				denominatorA[i][j] = elnsum(denominatorA[i][j], denominator);
			}

			for (k = 1; k <= hmm->M; k++) {
				numerator = NAN;
				denominator = NAN;
				for (t = 1; t <= T[l]; t++) {
					if (O[l][t] == k) {
						numerator = elnsum(numerator, gamma[t][i]);
					}
					denominator = elnsum(denominator, gamma[t][i]);
				}
				numeratorB[i][k] = elnsum(numeratorB[i][k], numerator);
				denominatorB[i][k] = elnsum(denominatorB[i][k], denominator);
			}
		}

		free_d3tensor(xi, 1, T[l], 1, hmm->N, 1, hmm->N);
	}

	/* reestimate frequency of state i in time t=1 */
	for (i = 1; i <= hmm->N; i++) {
		pisum = elnsum(pisum, logpi[i]);
	}

	for (i = 1; i <= hmm->N; i++) {
		hmm->pi[i] = elnproduct(logpi[i], -pisum);
	}

	/* reestimate transition matrix  and symbol prob in each state */
	for (i = 1; i <= hmm->N; i++) {
		for (j = 1; j <= hmm->N; j++) {
			hmm->A[i][j] = elnproduct(numeratorA[i][j], -denominatorA[i][j]);
		}
		for (k = 1; k <= hmm->M; k++) {
			hmm->B[i][k] = elnproduct(numeratorB[i][k], -denominatorB[i][k]);
		}
	}

	/* Calculate the probabilities with the newly estimated model */
	*plogprobinit = 0.0;
	for (l = 1; l <= L; l++) {
		dhmm_forward(hmm, T[l], O[l], alpha, &logprobf);
		*plogprobinit = elnproduct(*plogprobinit, logprobf); /* log P(O(L) |intial model) */
	}

	logprobprev = *plogprobinit;
	/* END first iteration */

	do {
		iter++;
		/* Initialize the numerators and denominators over l */
		pisum = NAN;
		for (i = 1; i <= hmm->N; i++) {
			logpi[i] = NAN;
			for (j = 1; j <= hmm->N; j++) {
				numeratorA[i][j] = NAN;
				denominatorA[i][j] = NAN;
			}
			for (k = 1; k <= hmm->M; k++) {
				numeratorB[i][k] = NAN;
				denominatorB[i][k] = NAN;
			}
		}

		for (l = 1; l <= L; l++) {
			dhmm_forward(hmm, T[l], O[l], alpha, &logprobf);
			dhmm_backward(hmm, T[l], O[l], beta, &logprobb);
			dhmm_compgamma(hmm, T[l], alpha, beta, gamma);

			xi = d3tensor(1, T[l], 1, hmm->N, 1, hmm->N);
			dhmm_compxi(hmm, T[l], O[l], alpha, beta, xi);

			for (i = 1; i <= hmm->N; i++) {
				logpi[i] = elnsum(logpi[i], gamma[1][i]);
				for (j = 1; j <= hmm->N; j++) {
					numerator = NAN;
					denominator = NAN;
					for (t = 1; t <= T[l] - 1; t++) {
						numerator = elnsum(numerator, xi[t][i][j]);
						denominator = elnsum(denominator, gamma[t][i]);
					}
					numeratorA[i][j] = elnsum(numeratorA[i][j], numerator);
					denominatorA[i][j] = elnsum(denominatorA[i][j], denominator);
				}

				for (k = 1; k <= hmm->M; k++) {
					numerator = NAN;
					denominator = NAN;
					for (t = 1; t <= T[l]; t++) {
						if (O[l][t] == k) {
							numerator = elnsum(numerator, gamma[t][i]);
						}
						denominator = elnsum(denominator, gamma[t][i]);
					}
					numeratorB[i][k] = elnsum(numeratorB[i][k], numerator);
					denominatorB[i][k] = elnsum(denominatorB[i][k], denominator);
				}
			}

			free_d3tensor(xi, 1, T[l], 1, hmm->N, 1, hmm->N);
		}

		/* reestimate frequency of state i in time t=1 */
		for (i = 1; i <= hmm->N; i++) {
			pisum = elnsum(pisum, logpi[i]);
		}

		for (i = 1; i <= hmm->N; i++) {
			hmm->pi[i] = elnproduct(logpi[i], -pisum);
		}

		/* reestimate transition matrix  and symbol prob in each state */
		for (i = 1; i <= hmm->N; i++) {
			for (j = 1; j <= hmm->N; j++) {
				hmm->A[i][j] = elnproduct(numeratorA[i][j], -denominatorA[i][j]);
			}
			for (k = 1; k <= hmm->M; k++) {
				hmm->B[i][k] = elnproduct(numeratorB[i][k], -denominatorB[i][k]);
			}
		}

		/* Calculate the probabilities with the newly estimated model */
		prodlogprob = 0.0;
		for (l = 1; l <= L; l++) {
			dhmm_forward(hmm, T[l], O[l], alpha, &logprobf);
			prodlogprob = elnproduct(prodlogprob, logprobf); /* log P(O(L) |intial model) */
		}

		/* compute difference between log probability of two iterations */
		if (isnan(prodlogprob)) {
			printf("prev: %lf\n", logprobprev);
			printf("curr: %lf\n", prodlogprob);
		}
		delta = prodlogprob - logprobprev;
		logprobprev = prodlogprob;

		printf("iteration %i: delta: %lf (%lf)\n", iter, delta, DELTA);
	} while ((delta > DELTA) && (iter < 100000)); /* if log probability does not change much, exit */

	/* Cleanup */
	free_dmatrix(numeratorA, 1, hmm->N, 1, hmm->N);
	free_dmatrix(denominatorA, 1, hmm->N, 1, hmm->N);
	free_dmatrix(numeratorB, 1, hmm->N, 1, hmm->M);
	free_dmatrix(denominatorB, 1, hmm->N, 1, hmm->M);

	*pniter = iter;
	*plogprobfinal = prodlogprob; /* log P(O(L)|estimated model) */
}

void dhmm_baumwelch_multiwt(DHMM *hmm, int *tied_obs, int *T, int L, int **O, double **alpha, double **beta, double **gamma, int *pniter, double *plogprobinit, double *plogprobfinal)
{
	int	i, j, k;
	int	t, iter = 1, l = 0;

	double	logprobf = NAN, logprobb;
	double prodlogprob = 0.0;

	double ***xi;
	double delta, logprobprev;

	double numerator;
	double denominator;
	double **numeratorA = dmatrix(1, hmm->N, 1, hmm->N);
	double **denominatorA = dmatrix(1, hmm->N, 1, hmm->N);
	double **numeratorB = dmatrix(1, hmm->N, 1, hmm->M);
	double **denominatorB = dmatrix(1, hmm->N, 1, hmm->M);

	double *logpi = dvector(1, hmm->N);
	double pisum = NAN;
	/* START first iteration */
	/* Initialize the numerators and denominators over l */
	pisum = NAN;
	for (i = 1; i <= hmm->N; i++) {
		logpi[i] = NAN;
		for (j = 1; j <= hmm->N; j++) {
			numeratorA[i][j] = NAN;
			denominatorA[i][j] = NAN;
		}
		for (k = 1; k <= hmm->M; k++) {
			numeratorB[i][k] = NAN;
			denominatorB[i][k] = NAN;
		}
	}

	for (l = 1; l <= L; l++) {
		dhmm_forward(hmm, T[l], O[l], alpha, &logprobf);
		dhmm_backward(hmm, T[l], O[l], beta, &logprobb);
		dhmm_compgamma(hmm, T[l], alpha, beta, gamma);

		xi = d3tensor(1, T[l], 1, hmm->N, 1, hmm->N);
		dhmm_compxi(hmm, T[l], O[l], alpha, beta, xi);

		for (i = 1; i <= hmm->N; i++) {
			logpi[i] = elnsum(logpi[i], gamma[1][i]);
			for (j = 1; j <= hmm->N; j++) {
				numerator = NAN;
				denominator = NAN;
				for (t = 1; t <= T[l] - 1; t++) {
					numerator = elnsum(numerator, xi[t][i][j]);
					denominator = elnsum(denominator, gamma[t][i]);
				}
				numeratorA[i][j] = elnsum(numeratorA[i][j], numerator);
				denominatorA[i][j] = elnsum(denominatorA[i][j], denominator);
			}

			for (k = 1; k <= hmm->M; k++) {
				numerator = NAN;
				denominator = NAN;
				for (t = 1; t <= T[l]; t++) {
					if (O[l][t] == k) {
						numerator = elnsum(numerator, gamma[t][i]);
					}
					denominator = elnsum(denominator, gamma[t][i]);
				}
				numeratorB[tied_obs[i]][k] = elnsum(numeratorB[tied_obs[i]][k], numerator);
				denominatorB[tied_obs[i]][k] = elnsum(denominatorB[tied_obs[i]][k], denominator);
			}
		}

		free_d3tensor(xi, 1, T[l], 1, hmm->N, 1, hmm->N);
	}

	/* reestimate frequency of state i in time t=1 */
	for (i = 1; i <= hmm->N; i++) {
		pisum = elnsum(pisum, logpi[i]);
	}

	for (i = 1; i <= hmm->N; i++) {
		hmm->pi[i] = elnproduct(logpi[i], -pisum);
	}

	/* reestimate transition matrix  and symbol prob in each state */
	for (i = 1; i <= hmm->N; i++) {
		for (j = 1; j <= hmm->N; j++) {
			hmm->A[i][j] = elnproduct(numeratorA[i][j], -denominatorA[i][j]);
		}
		for (k = 1; k <= hmm->M; k++) {
			hmm->B[i][k] = elnproduct(numeratorB[tied_obs[i]][k], -denominatorB[tied_obs[i]][k]);
		}
	}

	/* Calculate the probabilities with the newly estimated model */
	*plogprobinit = 0.0;
	for (l = 1; l <= L; l++) {
		dhmm_forward(hmm, T[l], O[l], alpha, &logprobf);
		*plogprobinit = elnproduct(*plogprobinit, logprobf); /* log P(O(L) |intial model) */
	}

	logprobprev = *plogprobinit;
	/* END first iteration */

	do {
		iter++;
		/* Initialize the numerators and denominators over l */
		pisum = NAN;
		for (i = 1; i <= hmm->N; i++) {
			logpi[i] = NAN;
			for (j = 1; j <= hmm->N; j++) {
				numeratorA[i][j] = NAN;
				denominatorA[i][j] = NAN;
			}
			for (k = 1; k <= hmm->M; k++) {
				numeratorB[i][k] = NAN;
				denominatorB[i][k] = NAN;
			}
		}

		for (l = 1; l <= L; l++) {
			dhmm_forward(hmm, T[l], O[l], alpha, &logprobf);
			dhmm_backward(hmm, T[l], O[l], beta, &logprobb);
			dhmm_compgamma(hmm, T[l], alpha, beta, gamma);

			xi = d3tensor(1, T[l], 1, hmm->N, 1, hmm->N);
			dhmm_compxi(hmm, T[l], O[l], alpha, beta, xi);

			for (i = 1; i <= hmm->N; i++) {
				logpi[i] = elnsum(logpi[i], gamma[1][i]);
				for (j = 1; j <= hmm->N; j++) {
					numerator = NAN;
					denominator = NAN;
					for (t = 1; t <= T[l] - 1; t++) {
						numerator = elnsum(numerator, xi[t][i][j]);
						denominator = elnsum(denominator, gamma[t][i]);
					}
					numeratorA[i][j] = elnsum(numeratorA[i][j], numerator);
					denominatorA[i][j] = elnsum(denominatorA[i][j], denominator);
				}

				for (k = 1; k <= hmm->M; k++) {
					numerator = NAN;
					denominator = NAN;
					for (t = 1; t <= T[l]; t++) {
						if (O[l][t] == k) {
							numerator = elnsum(numerator, gamma[t][i]);
						}
						denominator = elnsum(denominator, gamma[t][i]);
					}
					numeratorB[tied_obs[i]][k] = elnsum(numeratorB[tied_obs[i]][k], numerator);
					denominatorB[tied_obs[i]][k] = elnsum(denominatorB[tied_obs[i]][k], denominator);
				}
			}

			free_d3tensor(xi, 1, T[l], 1, hmm->N, 1, hmm->N);
		}

		/* reestimate frequency of state i in time t=1 */
		for (i = 1; i <= hmm->N; i++) {
			pisum = elnsum(pisum, logpi[i]);
		}

		for (i = 1; i <= hmm->N; i++) {
			hmm->pi[i] = elnproduct(logpi[i], -pisum);
		}

		/* reestimate transition matrix  and symbol prob in each state */
		for (i = 1; i <= hmm->N; i++) {
			for (j = 1; j <= hmm->N; j++) {
				hmm->A[i][j] = elnproduct(numeratorA[i][j], -denominatorA[i][j]);
			}
			for (k = 1; k <= hmm->M; k++) {
				hmm->B[i][k] = elnproduct(numeratorB[tied_obs[i]][k], -denominatorB[tied_obs[i]][k]);
			}
		}

		/* Calculate the probabilities with the newly estimated model */
		prodlogprob = 0.0;
		for (l = 1; l <= L; l++) {
			dhmm_forward(hmm, T[l], O[l], alpha, &logprobf);
			prodlogprob = elnproduct(prodlogprob, logprobf); /* log P(O(L) |intial model) */
		}

		/* compute difference between log probability of two iterations */
		if (isnan(prodlogprob)) {
			printf("prev: %lf\n", logprobprev);
			printf("curr: %lf\n", prodlogprob);
		}
		delta = prodlogprob - logprobprev;
		logprobprev = prodlogprob;

		printf("iteration %i: delta: %lf (%lf)\n", iter, delta, DELTA);
	} while ((delta > DELTA) && (iter < 100000)); /* if log probability does not change much, exit */

	/* Cleanup */
	free_dmatrix(numeratorA, 1, hmm->N, 1, hmm->N);
	free_dmatrix(denominatorA, 1, hmm->N, 1, hmm->N);
	free_dmatrix(numeratorB, 1, hmm->N, 1, hmm->M);
	free_dmatrix(denominatorB, 1, hmm->N, 1, hmm->M);

	*pniter = iter;
	*plogprobfinal = prodlogprob; /* log P(O(L)|estimated model) */
}

void dhmm_baumwelch_multi_constrained(DHMM *hmm, int *tied_obs, int *T, int L, int **O, char **cconstrmat, double **alpha, double **beta, double **gamma, int *pniter, double *plogprobinit, double *plogprobfinal)
{
	int	i, j, k;
	int	t, iter = 1, l = 0;

	double	logprobf = NAN, logprobb;
	double prodlogprob = 0.0;

	double ***xi;
	double delta, logprobprev;

	double numerator;
	double denominator;
	double **numeratorA = dmatrix(1, hmm->N, 1, hmm->N);
	double **denominatorA = dmatrix(1, hmm->N, 1, hmm->N);
	double **numeratorB = dmatrix(1, hmm->N, 1, hmm->M);
	double **denominatorB = dmatrix(1, hmm->N, 1, hmm->M);

	double *logpi = dvector(1, hmm->N);
	double pisum = NAN;
	/* START first iteration */
	/* Initialize the numerators and denominators over l */
	pisum = NAN;
	for (i = 1; i <= hmm->N; i++) {
		logpi[i] = NAN;
		for (j = 1; j <= hmm->N; j++) {
			numeratorA[i][j] = NAN;
			denominatorA[i][j] = NAN;
		}
		for (k = 1; k <= hmm->M; k++) {
			numeratorB[i][k] = NAN;
			denominatorB[i][k] = NAN;
		}
	}

	for (l = 1; l <= L; l++) {
		dhmm_forward(hmm, T[l], O[l], alpha, &logprobf);
		dhmm_backward(hmm, T[l], O[l], beta, &logprobb);
		dhmm_compgamma(hmm, T[l], alpha, beta, gamma);

		xi = d3tensor(1, T[l], 1, hmm->N, 1, hmm->N);
		dhmm_compxi(hmm, T[l], O[l], alpha, beta, xi);

		for (i = 1; i <= hmm->N; i++) {
			logpi[i] = elnsum(logpi[i], gamma[1][i]);
			for (j = 1; j <= hmm->N; j++) {
				numerator = NAN;
				denominator = NAN;
				for (t = 1; t <= T[l] - 1; t++) {
					/* if the name of the state we are in does not match, set the probability to 0 (NAN in eln) constrained
					 * we use a space to have unconstrained parts of the hidden states */
					if (hmm->nN[i] != cconstrmat[l - 1][t - 1]) {
						if (cconstrmat[l - 1][t - 1] != ' ') {
							xi[t][i][j] = NAN;
							gamma[t][i] = NAN;
						}
					}
					numerator = elnsum(numerator, xi[t][i][j]);
					denominator = elnsum(denominator, gamma[t][i]);
				}
				numeratorA[i][j] = elnsum(numeratorA[i][j], numerator);
				denominatorA[i][j] = elnsum(denominatorA[i][j], denominator);
			}

			for (k = 1; k <= hmm->M; k++) {
				numerator = NAN;
				denominator = NAN;
				for (t = 1; t <= T[l]; t++) {
					if (O[l][t] == k) {
						numerator = elnsum(numerator, gamma[t][i]);
					}
					denominator = elnsum(denominator, gamma[t][i]);
				}
				numeratorB[tied_obs[i]][k] = elnsum(numeratorB[tied_obs[i]][k], numerator);
				denominatorB[tied_obs[i]][k] = elnsum(denominatorB[tied_obs[i]][k], denominator);
			}
		}

		free_d3tensor(xi, 1, T[l], 1, hmm->N, 1, hmm->N);
	}

	/* reestimate frequency of state i in time t=1 */
	for (i = 1; i <= hmm->N; i++) {
		pisum = elnsum(pisum, logpi[i]);
	}

	for (i = 1; i <= hmm->N; i++) {
		hmm->pi[i] = elnproduct(logpi[i], -pisum);
	}

	/* reestimate transition matrix  and symbol prob in each state */
	for (i = 1; i <= hmm->N; i++) {
		for (j = 1; j <= hmm->N; j++) {
			hmm->A[i][j] = elnproduct(numeratorA[i][j], -denominatorA[i][j]);
		}
		for (k = 1; k <= hmm->M; k++) {
			hmm->B[i][k] = elnproduct(numeratorB[tied_obs[i]][k], -denominatorB[tied_obs[i]][k]);
		}
	}

	/* Calculate the probabilities with the newly estimated model */
	*plogprobinit = 0.0;
	for (l = 1; l <= L; l++) {
		dhmm_forward(hmm, T[l], O[l], alpha, &logprobf);
		*plogprobinit = elnproduct(*plogprobinit, logprobf); /* log P(O(L) |intial model) */
	}

	logprobprev = *plogprobinit;
	/* END first iteration */


	do {
		iter++;
		/* Initialize the numerators and denominators over l */
		pisum = NAN;
		for (i = 1; i <= hmm->N; i++) {
			logpi[i] = NAN;
			for (j = 1; j <= hmm->N; j++) {
				numeratorA[i][j] = NAN;
				denominatorA[i][j] = NAN;
			}
			for (k = 1; k <= hmm->M; k++) {
				numeratorB[i][k] = NAN;
				denominatorB[i][k] = NAN;
			}
		}

		for (l = 1; l <= L; l++) {
			dhmm_forward(hmm, T[l], O[l], alpha, &logprobf);
			dhmm_backward(hmm, T[l], O[l], beta, &logprobb);
			dhmm_compgamma(hmm, T[l], alpha, beta, gamma);

			xi = d3tensor(1, T[l], 1, hmm->N, 1, hmm->N);
			dhmm_compxi(hmm, T[l], O[l], alpha, beta, xi);

			for (i = 1; i <= hmm->N; i++) {
				logpi[i] = elnsum(logpi[i], gamma[1][i]);
				for (j = 1; j <= hmm->N; j++) {
					numerator = NAN;
					denominator = NAN;
					for (t = 1; t <= T[l] - 1; t++) {
						/* if the name of the state we are in does not match, set the probability to 0 (NAN in eln) constrained
						 * we use a space to have unconstrained parts of the hidden states */
						if (hmm->nN[i] != cconstrmat[l - 1][t - 1]) {
							if (cconstrmat[l - 1][t - 1] != ' ') {
								xi[t][i][j] = NAN;
								gamma[t][i] = NAN;
							}
						}
						numerator = elnsum(numerator, xi[t][i][j]);
						denominator = elnsum(denominator, gamma[t][i]);
					}
					numeratorA[i][j] = elnsum(numeratorA[i][j], numerator);
					denominatorA[i][j] = elnsum(denominatorA[i][j], denominator);
				}

				for (k = 1; k <= hmm->M; k++) {
					numerator = NAN;
					denominator = NAN;
					for (t = 1; t <= T[l]; t++) {
						if (O[l][t] == k) {
							numerator = elnsum(numerator, gamma[t][i]);
						}
						denominator = elnsum(denominator, gamma[t][i]);
					}
					numeratorB[tied_obs[i]][k] = elnsum(numeratorB[tied_obs[i]][k], numerator);
					denominatorB[tied_obs[i]][k] = elnsum(denominatorB[tied_obs[i]][k], denominator);
				}
			}

			free_d3tensor(xi, 1, T[l], 1, hmm->N, 1, hmm->N);
		}

		/* reestimate frequency of state i in time t=1 */
		for (i = 1; i <= hmm->N; i++) {
			pisum = elnsum(pisum, logpi[i]);
		}

		for (i = 1; i <= hmm->N; i++) {
			hmm->pi[i] = elnproduct(logpi[i], -pisum);
		}

		/* reestimate transition matrix  and symbol prob in each state */
		for (i = 1; i <= hmm->N; i++) {
			for (j = 1; j <= hmm->N; j++) {
				hmm->A[i][j] = elnproduct(numeratorA[i][j], -denominatorA[i][j]);
			}
			for (k = 1; k <= hmm->M; k++) {
				hmm->B[i][k] = elnproduct(numeratorB[tied_obs[i]][k], -denominatorB[tied_obs[i]][k]);
			}
		}

		/* Calculate the probabilities with the newly estimated model */
		prodlogprob = 0.0;
		for (l = 1; l <= L; l++) {
			dhmm_forward(hmm, T[l], O[l], alpha, &logprobf);
			prodlogprob = elnproduct(prodlogprob, logprobf); /* log P(O(L) |intial model) */
		}

		/* compute difference between log probability of two iterations */
		if (isnan(prodlogprob)) {
			printf("prev: %lf\n", logprobprev);
			printf("curr: %lf\n", prodlogprob);
		}
		delta = prodlogprob - logprobprev;
		logprobprev = prodlogprob;

		printf("iteration %i: delta: %lf (%lf)\n", iter, delta, DELTA);
	} while ((delta > DELTA) && (iter < 100000)); /* if log probability does not change much, exit */

	/* Cleanup */
	free_dmatrix(numeratorA, 1, hmm->N, 1, hmm->N);
	free_dmatrix(denominatorA, 1, hmm->N, 1, hmm->N);
	free_dmatrix(numeratorB, 1, hmm->N, 1, hmm->M);
	free_dmatrix(denominatorB, 1, hmm->N, 1, hmm->M);

	*pniter = iter;
	*plogprobfinal = prodlogprob; /* log P(O(L)|estimated model) */
}

/* --- Section with Continuous HMM --- */
/* Load a HMM from a file pointed to by filename */
void chmm_load(char *filename, CHMM *hmm)
{
	int i, j, k;
	FILE *fp = NULL;
	char *line = NULL;
	int len = 0;

	fp = fopen(filename, "r");
	if (fp == NULL) {
		fprintf(stderr, "Error: File %s not found\n", filename);
		exit(1);
	}

	line = hmm_fgetln(&line, &len, fp);
	if (strncmp("HMM type: Continuous", line, 20)) {
		fprintf(stderr, "Unsupported HMM format: %s\n", line);
		fprintf(stderr, "Only Discrete or Continuous supported!\n");
		exit(1);
	}
	free(line);

	fscanf(fp, "Number of states (N): %d\n", &(hmm->N));
	fscanf(fp, "Names of states:\n");
	hmm->nN = (char *) cvector(1, hmm->N);
	for (i = 1; i <= hmm->N; i++) {
		fscanf(fp, "%c ", &(hmm->nN[i]));
	}
	fscanf(fp, "\n");

	fscanf(fp, "Dimensions of observation vector (D): %d\n", &(hmm->D));
	fscanf(fp, "Names of observations:\n");
	hmm->nD = (char *) cvector(1, hmm->D);
	for (i = 1; i <= hmm->D; i++) {
		fscanf(fp, "%c ", &(hmm->nD[i]));
	}
	fscanf(fp, "\n");

	fscanf(fp, "Transition matrix (A):\n");
	hmm->A = (double **) dmatrix(1, hmm->N, 1, hmm->N);
	for (i = 1; i <= hmm->N; i++) {
		for (j = 1; j <= hmm->N; j++) {
			fscanf(fp, "%lf", &(hmm->A[i][j]));
			hmm->A[i][j] = eln(hmm->A[i][j]);
		}
		fscanf(fp, "\n");
	}

	fscanf(fp, "Emission means (B_miu):\n");
	hmm->B.miu = (double **) dmatrix(1, hmm->N, 1, hmm->D);
	for (j = 1; j <= hmm->N; j++) {
		for (k = 1; k <= hmm->D; k++) {
			fscanf(fp, "%lf", &(hmm->B.miu[j][k]));
		}
		fscanf(fp, "\n");
	}

	fscanf(fp, "Emission covariance (B_cov):\n");
	hmm->B.cov = (double **) dmatrix(1, hmm->N, 1, hmm->D);
	for (j = 1; j <= hmm->N; j++) {
		for (k = 1; k <= hmm->D; k++) {
			fscanf(fp, "%lf", &(hmm->B.cov[j][k]));
		}
		fscanf(fp, "\n");
	}

	fscanf(fp, "Initial state probability (pi):\n");
	hmm->pi = (double *) dvector(1, hmm->N);
	for (i = 1; i <= hmm->N; i++) {
		fscanf(fp, "%lf", &(hmm->pi[i]));
		hmm->pi[i] = eln(hmm->pi[i]);
	}

	/* Allocate and calculate inverse covariance */
	hmm->B.cov_inv = (double **) dmatrix(1, hmm->N, 1, hmm->D);
	for (j = 1; j <= hmm->N; j++) {
		for (k = 1; k <= hmm->D; k++) {
			hmm->B.cov_inv[j][k] = 1.0 / hmm->B.cov[j][k];
		}
	}

	fclose(fp);
}

void chmm_free(CHMM *hmm)
{
	free_cvector(hmm->nN, 1, hmm->N);
	free_cvector(hmm->nD, 1, hmm->D);
	free_dmatrix(hmm->A, 1, hmm->N, 1, hmm->N);
	free_dmatrix(hmm->B.miu, 1, hmm->N, 1, hmm->D);
	free_dmatrix(hmm->B.cov, 1, hmm->N, 1, hmm->D);
	free_dmatrix(hmm->B.cov_inv, 1, hmm->N, 1, hmm->D);
	free_dvector(hmm->pi, 1, hmm->N);
}

/* Initialize a HMM (A, B and pi) with random values */
void chmm_init(CHMM *hmm, int N, char *nN, int D, char *nD, int seed)
{
	int i, j, k;
	double sum;

	/* initialize random number generator */
	if (seed) {
		hmm_srand(seed);
	} else {
		hmm_srand(hmm_seed());
	}

	hmm->N = N;
	hmm->nN = (char *) cvector(1, hmm->N);
	for (i = 1; i <= hmm->N; i++) {
		hmm->nN[i] = nN[i];
	}

	hmm->D = D;
	hmm->nD = (char *) cvector(1, hmm->D);
	for (i = 1; i <= hmm->D; i++) {
		hmm->nD[i] = nD[i];
	}

	hmm->A = (double **) dmatrix(1, hmm->N, 1, hmm->N);
	for (i = 1; i <= hmm->N; i++) {
		sum = 0.0;

		for (j = 1; j <= hmm->N; j++) {
			hmm->A[i][j] = hmm_rand();
			sum += hmm->A[i][j];
		}

		for (j = 1; j <= hmm->N; j++) {
			hmm->A[i][j] /= sum;
		}
	}

	hmm->B.miu = (double **) dmatrix(1, hmm->N, 1, hmm->D);
	hmm->B.cov = (double **) dmatrix(1, hmm->N, 1, hmm->D);
	hmm->B.cov_inv = (double **) dmatrix(1, hmm->N, 1, hmm->D);
	for (j = 1; j <= hmm->N; j++) {
		for (k = 1; k <= hmm->D; k++) {
			hmm->B.miu[j][k] = hmm_rand();
			hmm->B.cov[j][k] = hmm_rand();
			hmm->B.cov_inv[j][k] = hmm_rand();
		}
	}

	hmm->pi = (double *) dvector(1, hmm->N);
	sum = 0.0;
	for (i = 1; i <= hmm->N; i++) {
		hmm->pi[i] = hmm_rand();
		sum += hmm->pi[i];
	}

	for (i = 1; i <= hmm->N; i++) {
		hmm->pi[i] /= sum;
	}
}

void chmm_copy(CHMM *hmm1, CHMM *hmm2)
{
	int i, j, k;

	hmm2->N = hmm1->N;
	hmm2->nN = (char *) cvector(1, hmm2->N);
	for (i = 1; i <= hmm2->N; i++) {
		hmm2->nN[i] = hmm1->nN[i];
	}

	hmm2->D = hmm1->D;
	hmm2->nD = (char *) cvector(1, hmm2->D);
	for (i = 1; i <= hmm2->D; i++) {
		hmm2->nD[i] = hmm1->nD[i];
	}

	hmm2->A = (double **) dmatrix(1, hmm2->N, 1, hmm2->N);

	for (i = 1; i <= hmm2->N; i++) {
		for (j = 1; j <= hmm2->N; j++) {
			hmm2->A[i][j] = hmm1->A[i][j];
		}
	}

	hmm2->B.miu = (double **) dmatrix(1, hmm2->N, 1, hmm2->D);
	hmm2->B.cov = (double **) dmatrix(1, hmm2->N, 1, hmm2->D);
	hmm2->B.cov_inv = (double **) dmatrix(1, hmm2->N, 1, hmm2->D);
	for (j = 1; j <= hmm2->N; j++) {
		for (k = 1; k <= hmm2->D; k++) {
			hmm2->B.miu[j][k] = hmm1->B.miu[j][k];
			hmm2->B.cov[j][k] = hmm1->B.cov[j][k];
			hmm2->B.cov_inv[j][k] = hmm1->B.cov_inv[j][k];
		}
	}

	hmm2->pi = (double *) dvector(1, hmm2->N);
	for (i = 1; i <= hmm2->N; i++) {
		hmm2->pi[i] = hmm1->pi[i];
	}
}

void chmm_save(char *filename, CHMM *hmm)
{
	int i, j, k;
	FILE *fp = NULL;
	int f = 1;

	if (!strcmp(filename, "stdout")) {
		fp = stdout;
		f = 0;
	} else {
		fp = fopen(filename, "w");
	}
	if (fp == NULL) {
		fprintf(stderr, "Error: File %s not found\n", filename);
		exit(1);
	}

	fprintf(fp, "HMM type: Continuous\n");

	fprintf(fp, "Number of states (N): %d\n", hmm->N);
	fprintf(fp, "Names of states:\n");
	for (i = 1; i <= hmm->N; i++) {
		fprintf(fp, "%c ", hmm->nN[i]);
	}
	fprintf(fp, "\n");

	fprintf(fp, "Dimensions of observation vector (D):%d\n", hmm->D);
	fprintf(fp, "Names of observations:\n");
	for (i = 1; i <= hmm->D; i++) {
		fprintf(fp, "%c ", hmm->nD[i]);
	}
	fprintf(fp, "\n");

	fprintf(fp, "Transition matrix (A):\n");
	for (i = 1; i <= hmm->N; i++) {
		for (j = 1; j <= hmm->N; j++) {
			fprintf(fp, "%.*e ", OP_DBL_Digs - 1, eexp(hmm->A[i][j]));
		}
		fprintf(fp, "\n");
	}

	fprintf(fp, "Emission means (B_miu):\n");
	for (j = 1; j <= hmm->N; j++) {
		for (k = 1; k <= hmm->D; k++) {
			fprintf(fp, "%.*e ", OP_DBL_Digs - 1, hmm->B.miu[j][k]);
		}
		fprintf(fp, "\n");
	}

	fprintf(fp, "Emission covariance (B_cov):\n");
	for (j = 1; j <= hmm->N; j++) {
		for (k = 1; k <= hmm->D; k++) {
			fprintf(fp, "%.*e ", OP_DBL_Digs - 1, hmm->B.cov[j][k]);
		}
		fprintf(fp, "\n");
	}

	fprintf(fp, "Initial state probability (pi):\n");
	for (i = 1; i <= hmm->N; i++) {
		fprintf(fp, "%.*e ", OP_DBL_Digs - 1, eexp(hmm->pi[i]));
	}
	fprintf(fp, "\n\n");

	if (f) {
		fclose(fp);
	}
}

struct samples *chmm_load_samples(char* filename)
{
	int i;
	int sample_count = 0;
	FILE *pf = fopen(filename, "rt");
	if (pf == NULL) {
		printf("Error open file!\n");
		return 0;
	}

	fscanf(pf, "sample_count= %d\n", &sample_count);

	struct samples * p_samples = (struct samples*)malloc(sizeof(struct samples));
	p_samples->data = 0;
	p_samples->sample_count = sample_count;
	p_samples->feature_count_per_sample = (int *)calloc(sample_count, sizeof(int));

	struct sample_node *sample_head = (struct sample_node *)malloc(sizeof(struct sample_node));
	LIST_HEAD_INIT(sample_head->list);

	p_samples->sample_head = sample_head;

	int count = 0;
	p_samples->feature_count_max = 0;
	int sample_index = 0;
	while (sample_index < sample_count) {
		fscanf(pf, "# %d\n", &count);
		p_samples->feature_count_per_sample[sample_index++] = count;
		if (count > p_samples->feature_count_max) {
			p_samples->feature_count_max = count;
		}
		struct sample_node *sample_tmp = (struct sample_node *)malloc(sizeof(sample_node));
		LIST_HEAD_INIT(sample_tmp->feature_head.list);
		while (count-- > 0) {
			struct feature_node *feature_tmp = (struct feature_node *)malloc(sizeof(struct feature_node));
			for (i = 0; i < DIM; i++) {
				fscanf(pf, "%lf", &feature_tmp->feature[i]);
			}
			list_add_tail(&feature_tmp->list, &sample_tmp->feature_head.list);
			fscanf(pf, "\n");
		}
		list_add_tail(&sample_tmp->list, &sample_head->list);
	}
	chmm_rebuild_samples(p_samples);
	return p_samples;
}

void chmm_rebuild_samples(struct samples* p_samples)
{
	p_samples->data = (double ***) calloc(p_samples->sample_count, sizeof(double**));
	struct sample_node *sample_head = p_samples->sample_head;

	struct list_head *pos_sample, *pos_feature;
	struct sample_node *p_sample;
	struct feature_node *p_feature;

	int sample_index = 0;
	list_for_each(pos_sample, &sample_head->list) {
		p_samples->data[sample_index] = (double**) calloc(p_samples->feature_count_per_sample[sample_index], sizeof(double*));
		p_sample = list_entry(pos_sample, struct sample_node, list);
		int feature_index = 0;
		list_for_each(pos_feature, &p_sample->feature_head.list) {
			p_feature = list_entry(pos_feature, struct feature_node, list);
			p_samples->data[sample_index][feature_index] = p_feature->feature;
			feature_index++;
		}
		sample_index++;
	}

}

void chmm_print_samples(struct samples * p_samples)
{
	int i = 0;
	struct sample_node *sample_head = p_samples->sample_head;

	struct list_head *pos_sample, *pos_feature;
	struct sample_node *p_sample;
	struct feature_node *p_feature;
	printf("sample_count= %d\n", p_samples->sample_count);

	int sample_index = 0;
	list_for_each(pos_sample, &sample_head->list) {
		printf("# %d\n", p_samples->feature_count_per_sample[sample_index++]);
		p_sample = list_entry(pos_sample, struct sample_node, list);
		list_for_each(pos_feature, &p_sample->feature_head.list) {
			p_feature = list_entry(pos_feature, struct feature_node, list);
			for (i = 0; i < DIM; i++) {
				printf("%f ", p_feature->feature[i]);
			}
			printf("\n");
		}
	}
}

/* Calc log outprobability after updating miu and cov */
void chmm_outprob(CHMM *hmm, double **sample, int T, double **outprob)
{
	int f, i, j;
	int N = hmm->N;
	int D = hmm->D;

	double *feature_tmp;

	double **miu = hmm->B.miu; // mean
	double **cov = hmm->B.cov; // covariance
	double **cov_inv = hmm->B.cov_inv;

	double prob1 = - 0.5 * D * eln(2.0 * M_PI); // -N*log(2pi)/2
	double *prob2 = (double*) dvector(1, N);  // -1/2*log(|sigma|)
	for (i = 1; i <= N; i++) {
		double tmp = 0.0;
		for (j = 1; j <= D; j++) {
			tmp += eln(cov[i][j]);
		}
		prob2[i] = - tmp * 0.5;
	}

	for (f = 0; f < T; f++) {
		double tmp, x;
		feature_tmp = sample[f];
		for (i = 1; i <= N; i++) {
			tmp = 0.0;
			for (j = 1; j <= D; j++) {
				x = feature_tmp[j - 1] - miu[i][j];
				tmp += x * x * cov_inv[i][j];
			}
			outprob[i][f + 1] = prob1 + prob2[i] - 0.5 * tmp;
		}
	}

	free_dvector(prob2, 1, N);
}

void chmm_forward(CHMM *hmm, int T, double **outprob, double **alpha, double *pprob)
{
	int     i, j;   /* state indices */
	int     t;      /* time index */

	double sum;     /* partial sum */

	/* 1. Initialization */

	for (i = 1; i <= hmm->N; i++) {
		alpha[1][i] = (hmm->pi[i] + outprob[i][1]);
	}

	/* 2. Induction */

	for (t = 1; t < T; t++) {
		for (j = 1; j <= hmm->N; j++) {
			sum = NAN;
			for (i = 1; i <= hmm->N; i++) {
				sum = elnsum(sum, elnproduct(alpha[t][i], hmm->A[i][j]));
			}
			alpha[t + 1][j] = (sum + outprob[j][t + 1]);
		}
	}

	/* 3. Termination */
	*pprob = NAN;
	for (i = 1; i <= hmm->N; i++) {
		*pprob = elnsum(*pprob, alpha[T][i]);
	}
}

void chmm_backward(CHMM *hmm, int T, double **outprob, double **beta, double *pprob)
{
	int     i, j;   /* state indices */
	int     t;      /* time index */
	double sum;


	/* 1. Initialization */

	for (i = 1; i <= hmm->N; i++) {
		beta[T][i] = 0.0;
	}

	/* 2. Induction */
	for (t = T - 1; t >= 1; t--) {
		for (i = 1; i <= hmm->N; i++) {
			sum = NAN;
			for (j = 1; j <= hmm->N; j++) {
				sum = elnsum(sum, elnproduct(elnproduct(hmm->A[i][j], outprob[j][t + 1]), beta[t + 1][j]));
			}
			beta[t][i] = sum;
		}
	}

	/* 3. Termination */
	*pprob = NAN;
	for (i = 1; i <= hmm->N; i++) {
		*pprob = elnsum(*pprob, elnproduct(hmm->pi[i], elnproduct(outprob[i][1], beta[1][i])));
	}
}

void chmm_viterbi(CHMM *hmm, int T, double **outprob, double **delta, int **psi, int *path, double *prob)
{
	int i, j; /* state indices */
	int t;    /* time index */

	int maxvalind;
	double maxval, val;

	/* Initialization  */
	for (i = 1; i <= hmm->N; i++) {
		delta[1][i] = elnproduct(hmm->pi[i], outprob[i][1]);
		psi[1][i] = 0;
	}

	/* Recursion */
	for (t = 2; t <= T; t++) {
		for (j = 1; j <= hmm->N; j++) {
			maxval = -DBL_MAX;
			maxvalind = 1;
			for (i = 1; i <= hmm->N; i++) {
				val = elnproduct(delta[t - 1][i], hmm->A[i][j]);
				if (val > maxval) {
					maxval = val;
					maxvalind = i;
				}
			}
			delta[t][j] = elnproduct(maxval, outprob[j][t]);
			psi[t][j] = maxvalind;
		}
	}

	/* Termination */
	*prob = -DBL_MAX;
	path[T] = 1;
	for (i = 1; i <= hmm->N; i++) {
		if (delta[T][i] > *prob) {
			*prob = delta[T][i];
			path[T] = i;
		}
	}

	/* Path (state sequence) backtracking */
	for (t = T - 1; t >= 1; t--) {
		path[t] = psi[t + 1][path[t + 1]];
	}
}

void chmm_compgamma(CHMM *hmm, int T, double **alpha, double **beta, double **gamma)
{
	int i; /* state indices */
	int	t; /* time index */
	double normalizer;

	for (t = 1; t <= T; t++) {
		normalizer = NAN;
		for (i = 1; i <= hmm->N; i++) {
			gamma[t][i] = elnproduct(alpha[t][i], beta[t][i]);
			normalizer = elnsum(normalizer, gamma[t][i]);
		}
		for (i = 1; i <= hmm->N; i++) {
			gamma[t][i] = elnproduct(gamma[t][i], -normalizer);
		}
	}
}

void chmm_compxi(CHMM *hmm, int T, double **outprob, double **alpha, double **beta, double ***xi)
{
	int i, j;
	int t;
	double normalizer;

	for (t = 1; t <= T - 1; t++) {
		normalizer = NAN;
		for (i = 1; i <= hmm->N; i++) {
			for (j = 1; j <= hmm->N; j++) {
				xi[t][i][j] = elnproduct(alpha[t][i], elnproduct(hmm->A[i][j], elnproduct(outprob[j][t + 1], beta[t + 1][j])));
				normalizer = elnsum(normalizer, xi[t][i][j]);
			}
		}
		for (i = 1; i <= hmm->N; i++) {
			for (j = 1; j <= hmm->N; j++) {
				xi[t][i][j] = elnproduct(xi[t][i][j], -normalizer);
			}
		}
	}
}

struct local_store_c {
	double **alpha;
	double **beta;
	double **outprob;
};

struct local_store_a {
	double **numeratorA;
	double **denominatorA;
	double **numeratorMiu;
	double **numeratorCov;
	double *denominatorM;
};

void f(CHMM *hmm, struct samples *p_samples, struct local_store_c *c, struct local_store_a *acc, double *logp)
{
	int i, j, k, t;
	int sample_index;

	int T;
	int D = hmm->D;
	int sample_count = p_samples->sample_count;
	int *fps = p_samples->feature_count_per_sample;

	double **alpha = c->alpha;
	double **beta = c->beta;
	double **outprob = c->outprob;

	double **sample;
	double logprobf = 0.0;
	double logprobb = 0.0;
	*logp = NAN;
	*logp = 0;
	int valid_count = 0;

	double ***xi;

	double **gamma = dmatrix(1, p_samples->feature_count_max, 1, hmm->N);

	double numerator;
	double denominator;

	double *logpi = dvector(1, hmm->N);
	double pisum = NAN;

	/* Initialize the numerators and denominators over l */
	pisum = NAN;
	for (i = 1; i <= hmm->N; i++) {
		logpi[i] = NAN;
		for (j = 1; j <= hmm->N; j++) {
			acc->numeratorA[i][j] = NAN;
			acc->denominatorA[i][j] = NAN;
		}
		for (j = 1; j <= hmm->D; j++) {
			acc->numeratorMiu[i][j] = 0.0;
			acc->numeratorCov[i][j] = 0.0;
		}
		acc->denominatorM[i] = 0.0;
	}

	for (sample_index = 0; sample_index < sample_count; sample_index++) {
		T = fps[sample_index];
		sample = p_samples->data[sample_index];
		chmm_outprob(hmm, sample, T, outprob);
		chmm_forward(hmm, T, outprob, alpha, &logprobf);
		chmm_backward(hmm, T, outprob, beta, &logprobb);

		valid_count++;

		*logp += logprobf;

		chmm_compgamma(hmm, T, alpha, beta, gamma);
		xi = d3tensor(1, T, 1, hmm->N, 1, hmm->N);
		chmm_compxi(hmm, T, outprob, alpha, beta, xi);

		for (i = 1; i <= hmm->N; i++) {
			logpi[i] = elnsum(logpi[i], gamma[1][i]);
			/* Accumulate A */
			for (j = 1; j <= hmm->N; j++) {
				numerator = NAN;
				denominator = NAN;
				for (t = 1; t <= T - 1; t++) {
					numerator = elnsum(numerator, xi[t][i][j]);
					denominator = elnsum(denominator, gamma[t][i]);
				}
				acc->numeratorA[i][j] = elnsum(acc->numeratorA[i][j], numerator);
				acc->denominatorA[i][j] = elnsum(acc->denominatorA[i][j], denominator);
			}
			/* Accumulate B*/
			for (t = 1; t <= T; t++) {
				double gamma_t_i = eexp(gamma[t][i]);
				for (k = 1; k <= D; k++) {
					acc->numeratorMiu[i][k] += gamma_t_i * sample[t - 1][k - 1];
					acc->numeratorCov[i][k] += gamma_t_i * (sample[t - 1][k - 1] - hmm->B.miu[i][k]) * (sample[t - 1][k - 1] - hmm->B.miu[i][k]);
				}
				acc->denominatorM[i] += gamma_t_i;
			}
		}

		free_d3tensor(xi, 1, T, 1, hmm->N, 1, hmm->N);
	} /* for sample*/

	/* reestimate frequency of state i in time t=1 */
	for (i = 1; i <= hmm->N; i++) {
		pisum = elnsum(pisum, logpi[i]);
	}

	for (i = 1; i <= hmm->N; i++) {
		hmm->pi[i] = elnproduct(logpi[i], -pisum);
	}

	/* reestimate transition matrix and observation prob in each state */
	for (i = 1; i <= hmm->N; i++) {
		/* update A */
		for (j = 1; j <= hmm->N; j++) {
			hmm->A[i][j] = elnproduct(acc->numeratorA[i][j], -acc->denominatorA[i][j]);
		}

		/* update B*/
		double c_d = 1.0 / acc->denominatorM[i];
		for (k = 1; k <= D; k++) {
			hmm->B.miu[i][k] = acc->numeratorMiu[i][k] * c_d;
			hmm->B.cov[i][k] = acc->numeratorCov[i][k] * c_d + MIN_COV;
			hmm->B.cov_inv[i][k] = 1.0 / hmm->B.cov[i][k];
		}
	}
	*logp /= valid_count;
}

void chmm_baumwelch(CHMM *hmm, struct samples *p_samples, int *piter, double *plogprobinit, double *plogprobfinal, int maxiter)
{
	int D = hmm->D;
	int N = hmm->N;
	int max_t = p_samples->feature_count_max;

	struct local_store_c cs;
	struct local_store_a as;

	double delta, logprob, logprobprev;

	cs.alpha = dmatrix(1, max_t, 1, N);
	cs.beta = dmatrix(1, max_t, 1, N);
	cs.outprob = dmatrix(1, hmm->N, 1, max_t);

	as.numeratorA = dmatrix(1, N, 1, N);
	as.denominatorA = dmatrix(1, N, 1, N);
	as.numeratorMiu = dmatrix(1, N, 1, D);
	as.numeratorCov = dmatrix(1, N, 1, D);
	as.denominatorM = dvector(1, N);

	logprobprev = -1000;

	*piter = 0;

	while (*piter < maxiter) {
		*piter = *piter + 1;
		/* compute difference between log probability of
			  two iterations */
		f(hmm, p_samples, &cs, &as, &logprob);

		delta = logprob - logprobprev;
		logprobprev = logprob;
		
		printf("iter %d, delta is : %.20f\n", *piter, delta);

		chmm_save("temp.hmm", hmm);

		if (fabs(delta) < DELTA) {
			break;
		}
	}
	//while (delta > DELTA);

	*plogprobfinal = logprob; /* log P(O|estimated model) */

	free_dmatrix(cs.alpha, 1, max_t, 1, N);
	free_dmatrix(cs.beta, 1, max_t, 1, N);
	free_dmatrix(cs.outprob, 1, hmm->N, 1, max_t);
	free_dmatrix(as.numeratorA, 1, N, 1, N);
	free_dmatrix(as.denominatorA, 1, N, 1, N);
	free_dmatrix(as.numeratorMiu, 1, N, 1, D);
	free_dmatrix(as.numeratorCov, 1, N, 1, D);
	free_dvector(as.denominatorM, 1, N);
}

int chmm_geninitialstate(CHMM *hmm);
int chmm_gennextstate(CHMM *hmm, int q_t);

void chmm_genseq(CHMM *hmm, int seed, int T, double *O, int *q)
{
	int t = 1;
	int i = 1;
	hmm_srand(seed);

	q[1] = chmm_geninitialstate(hmm);
	for (i = 1; i <= hmm->D; i++) {
		O[i] = hmm_norm_rand(hmm->B.miu[q[1]][i], sqrt(hmm->B.cov[q[1]][i]));
	}

	for (t = 2; t <= T; t++) {
		q[t] = chmm_gennextstate(hmm, q[t - 1]);
		for (i = 1; i <= hmm->D; i++) {
			O[((t - 1) * hmm->D) + i] = hmm_norm_rand(hmm->B.miu[q[t]][i], sqrt(hmm->B.cov[q[t]][i]));
		}
	}
}

int chmm_geninitialstate(CHMM *hmm)
{
	double value;
	double cummulative = 0.0;
	int i, q_t;

	value = hmm_rand();
	q_t = hmm->N;
	for (i = 1; i <= hmm->N; i++) {
		cummulative += eexp(hmm->pi[i]);
		if (value < cummulative) {
			q_t = i;
			break;
		}
	}

	return q_t;
}

int chmm_gennextstate(CHMM *hmm, int q_t)
{
	double value;
	double cummulative = 0.0;
	int j, q_next;

	value = hmm_rand();
	q_next = hmm->N;
	for (j = 1; j <= hmm->N; j++) {
		cummulative += eexp(hmm->A[q_t][j]);
		if (value < cummulative) {
			q_next = j;
			break;
		}
	}

	return q_next;
}

/* hmm_seed() generates an arbitary seed for the random number generator. */
uint64_t hmm_seed(void)
{
	return ((uint64_t) getpid() * (uint64_t) time(NULL));
}

/* hmm_srand() sets the seed of the random number generator to a specific value. */
void hmm_srand(uint64_t nseed)
{
	seed = nseed;
}

static inline uint32_t hmm_rotate32(const uint32_t x, int k)
{
	return (x << k) | (x >> (32 - k));
}

static inline double hmm_uint32_to_double(uint32_t x)
{
	const union {
		uint32_t i[2];
		double d;
	} u = { .i[0] = x >> 20, .i[1] = (x >> 12) | UINT32_C(0x3FF00000)};
	return u.d - 1.0;
}

/* hmm_rand() returns a (double) pseudo random number in the interval [0,1). */
double hmm_rand(void)
{
	unsigned int rotate = (unsigned int)(seed >> 59);
	seed = seed * 6364136223846793005u + (1442695040888963407u | 1u); /* LCG see: https://en.wikipedia.org/wiki/Linear_congruential_generator (Knuth) */
	seed ^= seed >> 18; /* XSH */
	hmm_rotate32((uint32_t)(seed >> 27), rotate); /* RR */

	return hmm_uint32_to_double(seed);
}

/* hmm_norm_rand() returns a (double) pseudo random number following a normal distribution with
 * a mean of mu and a stdev of sigma according to the Marsaglia and Bray method
 */
double hmm_norm_rand(double mu, double sigma)
{
	double U1, U2, W, mult;
	static double X1, X2;
	static int call = 0;

	/* if seed has not been initialized already, initialize it */
	if (seed == 0) {
		hmm_srand(hmm_seed());
	}

	if (call == 1) {
		call = !call;
		return (mu + sigma * (double) X2);
	}

	do {
		U1 = -1 + hmm_rand() * 2;
		U2 = -1 + hmm_rand() * 2;
		W = pow(U1, 2) + pow(U2, 2);
	} while (W >= 1 || W == 0);

	mult = sqrt((-2 * log(W)) / W);
	X1 = U1 * mult;
	X2 = U2 * mult;

	call = !call;

	return (mu + sigma * (double) X1);
}

/* Numerical Recipies Utility Functions - Public Domain */
void nrerror(const char *error_text)
{
	fprintf(stderr, "Numerical Recipes run-time error...\n");
	fprintf(stderr, "%s\n", error_text);
	fprintf(stderr, "...now exiting to system...\n");
	exit(1);
}

float *vector(int nl, int nh)
{
	float *v;

	v = (float *)calloc((unsigned)(nh - nl + 1), sizeof(float));
	if (!v) {
		nrerror("allocation failure in Vector()");
	}
	return v - nl;
}

int *ivector(int nl, int nh)
{
	int *v;

	v = (int *)calloc((unsigned)(nh - nl + 1), sizeof(int));
	if (!v) {
		nrerror("allocation failure in ivector()");
	}
	return v - nl;
}

double *dvector(int nl, int nh)
{
	double *v;

	v = (double *)calloc((unsigned)(nh - nl + 1), sizeof(double));
	if (!v) {
		nrerror("allocation failure in dvector()");
	}
	return v - nl;
}

char *cvector(int nl, int nh)
{
	char *v;

	v = (char *)calloc((unsigned)(nh - nl + 1), sizeof(char));
	if (!v) {
		nrerror("allocation failure in cvector()");
	}
	return v - nl;
}

float **matrix(int nrl, int nrh, int ncl, int nch)
{
	int i;
	float **m;

	m = (float **) calloc((unsigned)(nrh - nrl + 1), sizeof(float*));
	if (!m) {
		nrerror("allocation failure 1 in matrix()");
	}
	m -= nrl;

	for (i = nrl; i <= nrh; i++) {
		m[i] = (float *) calloc((unsigned)(nch - ncl + 1), sizeof(float));
		if (!m[i]) {
			nrerror("allocation failure 2 in matrix()");
		}
		m[i] -= ncl;


	}
	return m;
}

double **dmatrix(int nrl, int nrh, int ncl, int nch)
{
	int i;
	double **m;

	m = (double **) calloc((unsigned)(nrh - nrl + 1), sizeof(double*));
	if (!m) {
		nrerror("allocation failure 1 in dmatrix()");
	}
	m -= nrl;

	for (i = nrl; i <= nrh; i++) {
		m[i] = (double *) calloc((unsigned)(nch - ncl + 1), sizeof(double));
		if (!m[i]) {
			nrerror("allocation failure 2 in dmatrix()");
		}
		m[i] -= ncl;
	}
	return m;
}

double ***d3tensor(long nrl, long nrh, long ncl, long nch, long ndl, long ndh)
/* allocate a double 3tensor with range t[nrl..nrh][ncl..nch][ndl..ndh] */
{
	int i;
	double ***t;

	t = (double ***) malloc(nrh * sizeof(double **));

	t--;
	for (i = 1; i <= nrh; i++) {
		t[i] = dmatrix(ncl, nch, ndl, ndh);
	}

	return t;
}

int **imatrix(int nrl, int nrh, int ncl, int nch)
{
	int i, **m;

	m = (int **)calloc((unsigned)(nrh - nrl + 1), sizeof(int*));
	if (!m) {
		nrerror("allocation failure 1 in imatrix()");
	}
	m -= nrl;

	for (i = nrl; i <= nrh; i++) {
		m[i] = (int *)calloc((unsigned)(nch - ncl + 1), sizeof(int));
		if (!m[i]) {
			nrerror("allocation failure 2 in imatrix()");
		}
		m[i] -= ncl;
	}
	return m;
}

float **submatrix(float** a, int oldrl, int oldrh, int oldcl, int oldch, int newrl, int newcl)
{
	int i, j;
	float **m;

	m = (float **) calloc((unsigned)(oldrh - oldrl + 1), sizeof(float*));
	if (!m) {
		nrerror("allocation failure in submatrix()");
	}
	m -= newrl;

	for (i = oldrl, j = newrl; i <= oldrh; i++, j++) {
		m[j] = a[i] + oldcl - newcl;
	}

	return m;
}

void clear_dvector(double *m, int nl, int nh)
{
	int i;
	for (i = nl; i <= nh; i++) {
		m[i] = 0.0;
	}
}

void clear_dvector_nan(double *m, int nl, int nh)
{
	int i;
	for (i = nl; i <= nh; i++) {
		m[i] = NAN;
	}
}

void clear_imatrix(int **m, int  nrl, int nrh, int ncl, int nch)
{
	int i, j;
	for (i = nrl; i <= nrh; i++) {
		for (j = ncl; j <= nch; j++) {
			m[i][j] = 0;
		}
	}
}

void clear_dmatrix(double **m, int  nrl, int nrh, int ncl, int nch)
{
	int i, j;
	for (i = nrl; i <= nrh; i++) {
		for (j = ncl; j <= nch; j++) {
			m[i][j] = 0.0;
		}
	}
}

void clear_dmatrix_nan(double **m, int  nrl, int nrh, int ncl, int nch)
{
	int i, j;
	for (i = nrl; i <= nrh; i++) {
		for (j = ncl; j <= nch; j++) {
			m[i][j] = NAN;
		}
	}
}

void clear_dmatrix3d(double ***m, int nsl, int nsh, int  nrl, int nrh, int ncl, int nch)
{
	int i, j, k;
	for (i = nsl; i <= nsh; i++) {
		for (j = nrl; j <= nrh; j++)
			for (k = ncl; k <= nch; k++) {
				m[i][j][k] = 0.0;
			}
	}
}

void free_vector(float *v, int nl, int nh)
{
	free((char*)(v + nl));
}

void free_ivector(int *v, int nl, int nh)
{
	free((char*)(v + nl));
}

void free_dvector(double *v, int nl, int nh)
{
	free((char*)(v + nl));
}

void free_cvector(char *v, int nl, int nh)
{
	free((char*)(v + nl));
}

void free_matrix(float** m, int nrl, int nrh, int ncl, int nch)
{
	int i;

	for (i = nrh; i >= nrl; i--) {
		free((char*)(m[i] + ncl));
	}
	free((char*)(m + nrl));
}

void free_dmatrix(double **m, int nrl, int nrh, int ncl, int nch)
{
	int i;

	for (i = nrh; i >= nrl; i--) {
		free((char*)(m[i] + ncl));
	}
	free((char*)(m + nrl));
}

void free_d3tensor(double ***t, long nrl, long nrh, long ncl, long nch, long ndl, long ndh)
/* free a float d3tensor allocated by f3tensor() */
{
	int i;

	for (i = 1; i <= nrh; i++) {
		free_dmatrix(t[i], ncl, nch, ndl, ndh);
	}
	t++;

	free(t);
}

void free_imatrix(int **m, int nrl, int nrh, int ncl, int nch)
{
	int i;

	for (i = nrh; i >= nrl; i--) {
		free((char*)(m[i] + ncl));
	}
	free((char*)(m + nrl));
}

void free_submatrix(float **b, int nrl, int nrh, int ncl, int nch)
{
	free((char*)(b + nrl));
}

float **convert_matrix(float *a, int nrl, int nrh, int ncl, int nch)
{
	int i, j, nrow, ncol;
	float **m;

	nrow = nrh - nrl + 1;
	ncol = nch - ncl + 1;
	m = (float **) calloc((unsigned)(nrow), sizeof(float*));
	if (!m) {
		nrerror("allocation failure in convert_matrix()");
	}
	m -= nrl;
	for (i = 0, j = nrl; i <= nrow - 1; i++, j++) {
		m[j] = a + ncol * i - ncl;
	}
	return m;
}

void free_convert_matrix(float **b, int nrl, int nrh, int ncl, int nch)
{
	free((char*)(b + nrl));
}

/* Reads an entire file into an array of strings, needs only a single call to free */
char **hmm_fgetlns(char *filename, int *number_of_lines)
{
	size_t count = 1;
	char **sfile = NULL;
	char *p;

	FILE *f = NULL;
	f = fopen(filename, "r");
	if (!f) {
		fprintf(stderr, "Unable to open: %s\n", filename);
		return NULL;
	}

	/* Determine the file size */
	fseek(f, 0, SEEK_END);
	size_t fsize = ftell(f);
	fseek(f, 0, SEEK_SET);

	/* Read the file into a temporary buffer */
	char *buffer = malloc(fsize + 1);
	fread(buffer, fsize, 1, f);
	if (!buffer) {
		fprintf(stderr, "Failed to read %s into memory\n", filename);
		return NULL;
	}
	buffer[fsize] = 0;

	/* Close the file */
	fclose(f);

	/* Count the number of new lines */
	p = buffer;
	size_t i = 0;
	while (p[i]) {
		if (p[i] == '\n') {
			if (p[i + 1] == '\r') {
				count++;
				i++;
			} else {
				count++;
			}
		} else if (*p == '\r') {
			count++;
		}
		i++;
	}

	if (number_of_lines) {
		*number_of_lines = count;
	}

	/* Allocate space to keep the entire file */
	sfile = (char **) malloc(sizeof(char *) * (count + 1) + fsize + 1);
	if (!sfile) {
		fprintf(stderr, "Could not copy the data\n");
		return NULL;
	}
	sfile[count] = NULL;
	/* Copy in the original data */
	memcpy(&sfile[count + 1], buffer, fsize + 1);

	free(buffer);
	buffer = (char *) &sfile[count + 1];

	/* Go over everything again and set the pointers */
	p = buffer;
	i = 0;
	count = 0;
	sfile[count] = &p[i];
	while (p[i]) {
		if (p[i] == '\n') {
			if (p[i + 1] == '\r') {
				p[i] = '\0';
				p[i + 1] = '\0';
				count++;
				i++;
				if (p[i + 1]) {
					sfile[count] = &p[i + 1];
				}
			} else {
				p[i] = '\0';
				count++;
				if (p[i + 1]) {
					sfile[count] = &p[i + 1];
				}
			}
		} else if (*p == '\r') {
			p[i] = '\0';
			count++;
			if (p[i + 1]) {
				sfile[count] = &p[i + 1];
			}
		}
		i++;
	}

	return sfile;
}

/* Dynamic allocation version of fgets(), capable of reading "unlimited" line lengths. */
char *hmm_fgetln(char **buf, int *n, FILE *fp)
{
	char *s;
	int   len;
	int   location;

	/* Initial cache size */
	size_t cache_sz = 1024;
	/* Thereafter, each time capacity needs to be increased,
	 * multiply the increment by this factor.
	 */
	const size_t cache_sz_inc_factor = 2;

	if (*n == 0) {
		*buf = malloc(sizeof(char) * cache_sz);
		*n   = cache_sz;
	}

	/* We're sitting at EOF, or there's an error.*/
	if (fgets(*buf, *n, fp) == NULL) {
		return NULL;
	}

	/* We got a string AND it reached EOF. (last string without an '\n')*/
	if (feof(fp)) {
		return *buf;
	}

	/* We got a complete string */
	len = strlen(*buf);
	if ((*buf)[len - 1] == '\n') {
		return *buf;
	}

	/* We have an incomplete string and we have to extend the buffer.
	 * We make sure we overwrite the previous fgets \0 we got in the
	 * first step (and subsequent steps (location - 1) and append
	 * the new buffer until we find the \n or EOF.
	 */
	location = len;
	while (1) {
		*n  *= cache_sz_inc_factor;
		*buf = realloc(*buf, sizeof(char) * (*n));
		/* Append to previous buf */
		s = *buf + location;

		if (fgets(s, (*n - location), fp) == NULL) {
			return *buf;
		}

		if (feof(fp)) {
			return *buf;
		}

		len = strlen(s);
		if (s[len - 1] == '\n') {
			return *buf;
		}

		location = *n - 1;
	}
}

#endif
#endif
