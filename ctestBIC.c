#include <stdio.h> 
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define HMMTK_DEFINE
#include "hmmtk.h"

int main (int argc, char **argv)
{
	int 	 T; 
	CHMM  	 hmm;
	double **alpha;
	double 	 logproba;

	if (argc != 3) {
		printf("Usage error \n");
		printf("Usage: %s <model.hmm> <obs.seq> \n", argv[0]);
		exit (1);
	}

	chmm_load(argv[1], &hmm);

	struct samples *p_samples = chmm_load_samples(argv[2]);
	T = p_samples->feature_count_max;
	int max_t = T;
	double **outprob = dmatrix(1, hmm.N, 1, max_t);	

	double logprobtot = NAN;
	alpha = dmatrix(1, T, 1, hmm.N);

	double Ttot = 0;
	for (int i = 0; i < p_samples->sample_count; i++) {
		T = p_samples->feature_count_per_sample[i];
		/* length of observation series */
		Ttot += T;

		chmm_outprob(&hmm, p_samples->data[i], p_samples->feature_count_per_sample[i], outprob);
		/* Get probabilities */
		chmm_forward(&hmm, T, outprob, alpha, &logproba);

		/* Calcualte the total probability */
		logprobtot = elnsum(logprobtot, logproba);
	}

	int nullcount = 0;
	for (int i = 1; i <= hmm.N; i++) {
		for (int n = 1; n <= hmm.N; n++) {
			if (hmm.A[i][n] < DELTA) {
				nullcount++;
			}
		}
	}

	/* Simple version to determine the amount of free parameters */
	double p = hmm.N * (hmm.N - 1) - nullcount;

	/* Baysian Information Criterion
	 * AIC = -2 logL + p * logT 
	 */
	double BIC = (-2 * logprobtot) + (p * eln(Ttot));
	printf("BIC: %lf\n", BIC);

	/* Akaike information criterion
	 * AIC = -2 logL + 2p 
	 */
	double AIC = (-2 * logprobtot) + (2 * p);
	printf("AIC: %lf\n", AIC);

	free_dmatrix(alpha, 1, T, 1, hmm.N);
	
	chmm_free(&hmm);
}

