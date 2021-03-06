/*
**      Author: Tapas Kanungo, kanungo@cfar.umd.edu
**      Date:   22 February 1998 
**      File:   genseq.c
**      Purpose: driver for generating a sequence of observation symbols. 
**      Organization: University of Maryland
**
**	Update: 
**	Author: Tapas Kanungo
**	Purpose: randomize the seeds to generate random sequences
**		everytime the program is run.
**
**      $Id: genseq.c,v 1.4 1999/05/04 15:36:53 kanungo Exp kanungo $
*/

#include <stdio.h> 
#include <stdlib.h> 
#include <math.h>
#include <string.h>
#define HMMTK_DEFINE
#include "hmmtk.h"
#include <sys/types.h>
#include <unistd.h> 

void Usage(char *name);

int main (int argc, char **argv)
{
	CHMM  	hmm; 	/* the HMM */
	int  	T = 100; 	/* length of observation sequence */
	double	*O;	/* the observation sequence O[1..T]*/
	int	*q; 	/* the state sequence q[1..T] */
	int	sflg=0, tflg=0, errflg = 0;
	int	seed;	/* random number seed */
	int	c;	
	extern char *optarg;
	extern int optind, opterr, optopt;



	while ((c= getopt(argc, argv, "S:T:")) != EOF)
		switch (c) {
		case 'S':
			/* set random number generator seed */
			if (sflg)
				errflg++;
			else {	
				sflg++;
				sscanf(optarg, "%d", &seed);
			}
			break;
		case 'T':
			/* set sequence length */
			if (tflg)
				errflg++;
			else {	
				tflg++;
				sscanf(optarg, "%d", &T);
			}
			break;
		case '?':
			errflg++;
		}

	if ((argc - optind) != 1) errflg++; /* number or arguments not
					       okay */

	if (errflg || !tflg ) {
		Usage(argv[0]);
		exit(1);
	}

	chmm_load(argv[optind], &hmm);

	/* length of observation sequence, T */
	O = dvector(1, T * hmm.D); /* alloc space for observation sequence O */
	q = ivector(1, T); /* alloc space for state sequence q */

	if (!sflg) seed = hmm_seed();

	fprintf(stderr, "RandomSeed: %d\n", seed);
	chmm_genseq(&hmm, seed, T, O, q);

	printf("Observed symbols:\n");
	for (int i = 1; i <= T * hmm.D; i++) {
		printf("%lf ", O[i]);
		if (!(i % hmm.D)) {
			printf("\n");
		}
	}

	printf("Hidden states:\n");
	for (int i = 1; i <= T; i++) {
		printf("%d\n", q[i]);
	}

	free_dvector(O, 1, T * hmm.D);
	free_ivector(q, 1, T);
	chmm_free(&hmm);
}

void Usage(char *name)
{
    printf("Usage error \n");
    printf("Usage: %s -T <sequence length> <mod.hmm> \n", name);
    printf("Usage: %s -S <seed> -T <sequence length> <mod.hmm> \n",
            name);
    printf("  T = length of sequence\n");
    printf("  S =  random number seed \n");
    printf("  mod.hmm is a file with HMM parameters\n");
}
