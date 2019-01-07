/*
**      Author: Tapas Kanungo, kanungo@cfar.umd.edu
**      Date:   4 May 1999 
**      File:   testfor.c
**      Purpose: driver for testing the Forward, ForwardWithScale code.
**      Organization: University of Maryland
**
**	$Id$
*/

#include <stdio.h> 
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define HMMTK_DEFINE
#include "hmmtk.h"

int main (int argc, char **argv)
{
	int      T; 
	DHMM      hmm;
	int	    *O;	/* observation sequence O[1..T] */
	double **alpha;
	double **beta;
	double 	 logproba = 0.0; 

	if (argc != 3) {
		printf("Usage error \n");
		printf("Usage: testfor <model.hmm> <obs.seq> \n");
		exit (1);
	}

	dhmm_load(argv[1], &hmm);
	dhmm_loadseq(argv[2], &T, &O);

	alpha = dmatrix(1, T, 1, hmm.N);
	dhmm_forward(&hmm, T, O, alpha, &logproba);
	fprintf(stdout, "Forward log prob(observations | model) = %E\n", logproba);
	
	beta = dmatrix(1, T, 1, hmm.N);
	dhmm_backward(&hmm, T, O, beta, &logproba);
	fprintf(stdout, "Backward log prob(observations | model) = %E\n", logproba);

	free_dmatrix(alpha, 1, T, 1, hmm.N);
	free_dmatrix(beta, 1, T, 1, hmm.N);
	free_ivector(O, 1, T);
	dhmm_free(&hmm);
}

