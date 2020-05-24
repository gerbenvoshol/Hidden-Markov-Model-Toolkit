#include <stdio.h> 
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define HMMTK_DEFINE
#include "../../hmmtk.h"

#define SEQIO_DEFINE
#include "../../seqio.h"

int lookup(char c, char *alphabet)
{
	int i;

	for (i = 1; i < strlen(alphabet); i++) {
		if (c == alphabet[i]) {
			return i;
		}
	}

	fprintf(stderr, "Could not find \"%c\" in HMM symbols\n", c);
	return 0;
}

int main (int argc, char **argv)
{
	int 	 T; 
	DHMM  	 hmm;
	double	    **O;	/* observation sequence O[1..T] */
	double **alpha;
	double **beta;
	double	**gamma;
	double **state_prob;
	double 	 logproba; 
	int *path;
	double maxval, val;
	int maxvalind;

	if (argc != 3) {
		printf("Usage error \n");
		printf("Usage: %s <model.hmm> <obs.seq> \n", argv[0]);
		exit (1);
	}

	dhmm_load(argv[1], &hmm);

	char states[5] = "0122H";


	SEQFILE *sqfp;
	char *sequence;
	char *description;
	sqfp = seqfopen(argv[2], "r", NULL);
	while ((sequence = seqfgetseq(sqfp, NULL, 1)) != NULL) {
		description = seqfdescription(sqfp, 0);
		dhmm_cstr2csprof(&hmm, line[j+1], &T, &O);

		path = ivector(1, T);

		alpha = dmatrix(1, T, 1, hmm.N);
		dhmm_forward_prof(&hmm, T, O, alpha, &logproba);
		
		beta = dmatrix(1, T, 1, hmm.N);
		dhmm_backward_prof(&hmm, T, O, beta, &logproba);

		gamma = dmatrix(1, T, 1, hmm.N);
		dhmm_compgamma(&hmm, T, alpha, beta, gamma);

		state_prob = dmatrix(1, T, 1, 5);
		for (int i = 1; i <= T; i++) {
			for (int n = 1; n <= 5; n++) {
				state_prob[i][n] = NAN;
			}
		}

		for (int i = 1; i <= T; i++) {
			maxval = gamma[i][1];
            maxvalind = 1;
			for (int n = 1; n <= hmm.N; n++) {
            	val = gamma[i][n];
            	if (val > maxval) {
            	    maxval = val;
            	    maxvalind = n;
            	}
				state_prob[i][lookup(hmm.nN[n], states)] = elnsum(state_prob[i][lookup(hmm.nN[n], states)], gamma[i][n]);
			}
			path[i] = maxvalind;
		}

		for (int i = 1; i <= T; i++) {
			maxval = eexp(state_prob[i][1]);
            maxvalind = 1;
			for (int n = 1; n <= 5; n++) {
				val = eexp(state_prob[i][n]);
            	if (val > maxval) {
            	    maxval = val;
            	    maxvalind = n;
            	}
			}
			path[i] = maxvalind;
		}
		
		printf("%s\n", description);
		printf("%s\n", sequence);
		for (int i = 1; i <= T; i++) {
			printf("%c", states[path[i]]);
		}
		printf("\n");

		free(sequence);

		free_ivector(path, 1, T);
		
		free_dmatrix(alpha, 1, T, 1, hmm.N);
		free_dmatrix(beta, 1, T, 1, hmm.N);
		free_dmatrix(gamma, 1, T, 1, hmm.N);

	}
	
	seqfclose(sqfp);

	dhmm_free(&hmm);
}

