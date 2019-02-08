/* Baum-Welch training with Tied Observation Emission probabilities */
/* ./trainconstr SIGNALPEPTIDE_LABELLED.hmm all.training.labelled.data all.training.constr */
#include <stdio.h> 
#include <stdlib.h>
#include <math.h>
#include <string.h>
#define HMMTK_DEFINE
#include "hmmtk.h"
#include "seqio.h"

#include <sys/types.h>
#include <unistd.h>

void Usage(char *name);

int main (int argc, char **argv)
{
	int 	*T;
	int     L;
	DHMM  	hmm;
	double 	**alpha; 
	double	**beta;
	double	**gamma;
	int	**O;
	int	niter;
	double	logprobinit, logprobfinal;

	if (argc < 4) {
		Usage(argv[0]);
		exit (1);
	}
	
	printf("Loading initial HMM");
	/* Load the HMM */
	dhmm_load(argv[1], &hmm);
	printf("...done!\n");

	printf("Reading sequences");
	/* Read the observed sequences */
	char **sequences = hmm_fgetlns(argv[2], &L);
	printf("...done!\n");

	printf("Reading labels");
	/* Read the constrains */
	char **constrains = hmm_fgetlns(argv[3], &L);
	printf("...done!\n");

	printf("Converting sequences to observations");
	int maxT = dhmm_cstrmat2seq(&hmm, sequences, &T, &L, &O);
	printf("...done\n");

	/* allocate memory */
	alpha = dmatrix(1, maxT, 1, hmm.N);
	beta = dmatrix(1, maxT, 1, hmm.N);
	gamma = dmatrix(1, maxT, 1, hmm.N);

	/* print the answer */
	//printf("Initial model:\n");
	//dhmm_save("stdout", &hmm);

	/* Tied observational emission states 
	 * For example observation state 1 (Tied_Obs[1]) has its own state (1), While 2
	 * Tied_Obs[2] has its own state, and 3 (Tied_Obs[3]) has the same emission probabilities as 2. This 
	 * continues til we reach emission state 9 with another state which is share by 20 states after that.
	 */
	// With tied H domain
	//int Tied_Obs[110] = {0, 1, 2, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 2, 1, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 50, 50, 50, 50, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 2, 2, 1, 4, 4, 4, 4, 4, 72, 73, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 50, 50, 50, 50, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 2, 2};
	// 2 residues after RR are free to train RR.XX
	//int Tied_Obs[110] = {0, 1, 2, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 2, 1, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 50, 50, 50, 50, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 2, 2, 1, 4, 4, 4, 4, 4, 72, 73, 5, 75, 76, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 50, 50, 50, 50, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 2, 2};
	// sec and tat independent n and h region
	// int Tied_Obs[110] = {0, 1, 2, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 2, 1, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 50, 50, 50, 50, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 2, 2, 1, 67, 67, 67, 67, 67, 72, 73, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 50, 50, 50, 50, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 2, 2};
	// 1, 2, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 2, 1, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 50, 50, 50, 50, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 2, 2, 1, 67, 67, 67, 67, 67, 72, 73, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 94, 94, 94, 94, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 2, 2
	// sec and tat independent n, h and intial part of c region (except cleavage site)
	//int Tied_Obs[110] = {0, 1, 2, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 2, 1, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 50, 50, 50, 50, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 2, 2, 1, 67, 67, 67, 67, 67, 72, 73, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 94, 94, 94, 94, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 2, 2};
	// sec and tat independent n, h and intial part of c region (except cleavage site) and untied residues around XRR.XX
	//int Tied_Obs[110] = {0, 1, 2, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 2, 1, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 50, 50, 50, 50, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 2, 2, 1, 67, 67, 67, 67, 71, 72, 73, 74, 75, 76, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 94, 94, 94, 94, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 2, 2};
	// sec and tat independent n, h and intial part of c region (except cleavage site) and untied residues around XXRR.XXX
	//int Tied_Obs[110] = {0, 1, 2, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 2, 1, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 50, 50, 50, 50, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 2, 2, 1, 67, 67, 67, 70, 71, 72, 73, 74, 75, 76, 77, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 94, 94, 94, 94, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 2, 2};
	// sec and tat independent n, h and intial part of c region (except cleavage site) and untied residues around XXRRXXXX
	int Tied_Obs[110] = {0, 1, 2, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 2, 1, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 50, 50, 50, 50, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 2, 2, 1, 67, 67, 67, 70, 71, 72, 73, 74, 75, 76, 77, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 94, 94, 94, 94, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 2, 2};
	 // sec and tat independent n, h and c region and untied residues around XXRRXXXX
	//int Tied_Obs[110] = {0, 1, 2, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 2, 1, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 50, 50, 50, 50, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 2, 2, 1, 67, 67, 67, 70, 71, 72, 73, 74, 75, 76, 77, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 94, 94, 94, 94, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 2, 2};
	/* call Baum Welch with Tied Observations probabilities */
	dhmm_baumwelch_multi_constrained(&hmm, Tied_Obs, T, L, O, constrains, alpha, beta, gamma, &niter, &logprobinit, &logprobfinal);

	/* free memory */
	free_imatrix(O, 1, L, 1, maxT);
	free_ivector(T, 1, L);
	free_dmatrix(alpha, 1, maxT, 1, hmm.N);
	free_dmatrix(beta, 1, maxT, 1, hmm.N);
	free_dmatrix(gamma, 1, maxT, 1, hmm.N);
	free(sequences);
	free(constrains);

	/* print the answer */
	dhmm_save("stdout", &hmm);

	dhmm_free(&hmm);
}

void Usage(char *name)
{
	printf("Usage3: %s [-v] -I <mod.hmm> <file.seq> <constrains>\n", name); 
}
