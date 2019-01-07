/*
**      Author: Tapas Kanungo, kanungo@cfar.umd.edu
**      Date:   15 December 1997
**      File:   testvit.c
**      Purpose: driver for testing the Viterbi code.
**      Organization: University of Maryland
**
**	Update:
**	Author:	Tapas Kanungo
**	Purpose: run both viterbi with probabilities and 
**		viterbi with log, change output etc.
**      $Id: testvit.c,v 1.3 1998/02/23 07:39:07 kanungo Exp kanungo $
*/

#include <stdio.h> 
#include <stdlib.h>
#include <math.h>
#include <string.h>
#define HMMTK_DEFINE
#include "hmmtk.h"

int main (int argc, char **argv)
{
	CHMM  	 hmm;
	int	    *q;	/* state sequence q[1..T] */
	double **delta;
	int	   **psi;
	double 	 logproba; 

	if (argc != 3) {
		printf("Usage error \n");
		printf("Usage: testvit <model.hmm> <obs.seq> \n");
		exit (1);
	}

	chmm_load(argv[1], &hmm);

	struct samples *p_samples = chmm_load_samples(argv[2]);
	int max_t = p_samples->feature_count_max;
	double **outprob = dmatrix(1, hmm.N, 1, max_t);	
	//chmm_print_samples(p_samples);
	chmm_outprob(&hmm, p_samples->data[0], p_samples->feature_count_per_sample[0], outprob);

	q = ivector(1,max_t);
	delta = dmatrix(1, max_t, 1, hmm.N);
	psi = imatrix(1, max_t, 1, hmm.N);

	printf("------------------------------------\n");
	printf("Viterbi using log probabilities\n");

	chmm_viterbi(&hmm, max_t, outprob, delta, psi, q, &logproba); 
	fprintf(stdout, "Viterbi MLE log prob = %E\n", logproba);
	fprintf(stdout, "Optimal state sequence:\n");
	
	dhmm_saveseq("stdout", max_t, q);

	free_ivector(q, 1, max_t);
	free_imatrix(psi, 1, max_t, 1, hmm.N);
	free_dmatrix(delta, 1, max_t, 1, hmm.N);
	
	chmm_free(&hmm);
}

