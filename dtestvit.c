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
	int 	 T; 
	DHMM  	 hmm;
	int	    *O;	/* observation sequence O[1..T] */
	int	    *q;	/* state sequence q[1..T] */
	double **delta;
	int	   **psi;
	double 	 logproba; 

	if (argc != 3) {
		printf("Usage error \n");
		printf("Usage: testvit <model.hmm> <obs.seq> \n");
		exit (1);
	}

	dhmm_load(argv[1], &hmm);

	dhmm_loadseq(argv[2], &T, &O);

	q = ivector(1,T);
	delta = dmatrix(1, T, 1, hmm.N);
	psi = imatrix(1, T, 1, hmm.N);

	printf("------------------------------------\n");
	printf("Viterbi using log probabilities\n");

	dhmm_viterbi(&hmm, T, O, delta, psi, q, &logproba); 
	fprintf(stdout, "Viterbi MLE log prob = %E\n", logproba);
	fprintf(stdout, "Optimal state sequence:\n");
	
	dhmm_saveseq("stdout", T, q);

	free_ivector(q, 1, T);
	free_ivector(O, 1, T);
	free_imatrix(psi, 1, T, 1, hmm.N);
	free_dmatrix(delta, 1, T, 1, hmm.N);
	
	dhmm_free(&hmm);
}

