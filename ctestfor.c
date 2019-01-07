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
	int      T = 0; 
	CHMM      hmm;
	double **alpha = NULL;
	double **beta = NULL;
	double 	 logproba = 0.0;
	double 	 logprobb = 0.0; 

	if (argc != 3) {
		printf("Usage error \n");
		printf("Usage: testfor <model.hmm> <obs.seq> \n");
		exit (1);
	}

	chmm_load(argv[1], &hmm);
	struct samples *p_samples = chmm_load_samples(argv[2]);

	//chmm_save("stdout", &hmm);

	int max_t = p_samples->feature_count_max;
	alpha = dmatrix(1, max_t, 1, hmm.N);
	double **outprob = dmatrix(1, hmm.N, 1, max_t);	

	chmm_outprob(&hmm, p_samples->data[0], p_samples->feature_count_per_sample[0], outprob);

	beta = dmatrix(1, max_t, 1, hmm.N);
	chmm_backward(&hmm, max_t, outprob, beta, &logprobb);
	fprintf(stdout, "Backward log prob(observations | model) = %E\n", logprobb);

	chmm_forward(&hmm, p_samples->feature_count_per_sample[0], outprob, alpha, &logproba);
	fprintf(stdout, "Forward log prob(observations | model) = %E\n", logproba);

	free_dmatrix(alpha, 1, T, 1, hmm.N);
	free_dmatrix(beta, 1, T, 1, hmm.N);
	chmm_free(&hmm);
}

