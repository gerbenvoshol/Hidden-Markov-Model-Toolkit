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
	int	    *O;	/* observation sequence O[1..T] */
	double **alpha;
	double **beta;
	double **gamma;
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

	SEQFILE *sqfp;
	char *sequence;
	char *description;
	sqfp = seqfopen(argv[2], "r", NULL);
	char states[4] = "0HcE";
	while ((sequence = seqfgetseq(sqfp, NULL, 1)) != NULL) {
		description = seqfdescription(sqfp, 0);

		printf(">%s\n", description);
		printf("%s\n", sequence);

		dhmm_cstr2seq(&hmm, sequence, &T, &O);

		free(sequence);

		path = ivector(1, T);

		alpha = dmatrix(1, T, 1, hmm.N);
		dhmm_forward(&hmm, T, O, alpha, &logproba);
		// fprintf(stdout, "Forward log prob(observations | model) = %E\n", logproba);
		
		beta = dmatrix(1, T, 1, hmm.N);
		dhmm_backward(&hmm, T, O, beta, &logproba);
		// fprintf(stdout, "Backward log prob(observations | model) = %E\n", logproba);

		gamma = dmatrix(1, T, 1, hmm.N);
		dhmm_compgamma(&hmm, T, alpha, beta, gamma);

		state_prob = dmatrix(1, T, 1, 3);
		for (int i = 1; i <= T; i++) {
			for (int n = 1; n <= 3; n++) {
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

		//printf("PostProb\n");
		//printf("H\tc\tE\n");
		for (int i = 1; i <= T; i++) {
			maxval = gamma[i][1];
            maxvalind = 1;
			for (int n = 1; n <= 3; n++) {
				//printf("%lf\t", state_prob[i][n]);
				val = state_prob[i][n];
            	if (val > maxval) {
            	    maxval = val;
            	    maxvalind = n;
            	}
			}
			path[i] = maxvalind;
			//printf("\n");
		}
		
		for (int i = 1; i <= T; i++) {
			printf("%c", states[path[i]]);
		}
		printf("\n");

		free_ivector(path, 1, T);
		
		free_dmatrix(alpha, 1, T, 1, hmm.N);
		free_dmatrix(beta, 1, T, 1, hmm.N);
		free_dmatrix(gamma, 1, T, 1, hmm.N);

	}
	
	dhmm_free(&hmm);
}

