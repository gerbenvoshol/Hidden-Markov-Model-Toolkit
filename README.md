# Hidden-Markov-Model-Toolkit
Simple discrete (DHMM) and continuous Hidden Markov Model (CHMM) library

 ============================================================================
	 
	 You MUST

			#define HMMTK_DEFINE

	 in EXACTLY _one_ C or C++ file that includes this header, BEFORE the
	 include, like this:

			#define HMMTK_DEFINE
			#include "hmmtk.h"

	 All other files should just #include "hmmtk.h" without the #define.
	 
 ============================================================================

 Version History
  		1.2   Added labelled/constrained Baum-Welch training
 		1.1   Fixed memory leak, fixed the chmm_outprob() function (= P(x|mu, var)), all 
 		      values of A, B and pi are now stored as ln() internally instead of converting 
 		      them in the HMM functions
		1.0   Initial release containing basic discrete and continuous hmm functions

 CITATION

 If you use this HMM Toolkit in a publication, please reference:
 
 Voshol, G.P. (2020). HMMTK: A simple HMM Toolkit (Version 1.2) [Software]. 
 Available from https://github.com/gerbenvoshol/Hidden-Markov-Model-Toolkit
