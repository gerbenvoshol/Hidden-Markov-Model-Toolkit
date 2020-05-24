#include <stdio.h> 
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define HMMTK_DEFINE
#include "../../hmmtk.h"

#define SEQIO_DEFINE
#include "../../seqio.h"

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
		printf("Usage: %s <model.hmm> <obs.seq> \n", argv[0]);
		exit (1);
	}

	dhmm_load(argv[1], &hmm);

	SEQFILE *sqfp;
	char *sequence;
	char *description;
	sqfp = seqfopen(argv[2], "r", NULL);
	while ((sequence = seqfgetseq(sqfp, NULL, 1)) != NULL) {
		description = seqfdescription(sqfp, 0);

		/* Truncate sequence */
		sequence[70] = '\0';

		dhmm_cstr2seq(&hmm, sequence, &T, &O);

		free(sequence);

		q = ivector(1,T);
		delta = dmatrix(1, T, 1, hmm.N);
		psi = imatrix(1, T, 1, hmm.N);

		dhmm_viterbi(&hmm, T, O, delta, psi, q, &logproba); 

		printf(">%s", description);
		fprintf(stdout, " (Viterbi MLE log prob = %E)\n", logproba);
		printf("%s\n", sequence);
		for (int i = 1; i <= T; i++) {
			printf("%c", hmm.nN[q[i]]);
		}
		printf("\n");

		free_ivector(q, 1, T);
		free_ivector(O, 1, T);
		free_imatrix(psi, 1, T, 1, hmm.N);
		free_dmatrix(delta, 1, T, 1, hmm.N);
	}
	
	seqfclose(sqfp);
	
	dhmm_free(&hmm);
}

