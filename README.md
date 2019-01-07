# Hidden-Markov-Model-Toolkit
Simple discrete (DHMM) and continuous Hidden Markov Model (CHMM) library
	 This is a single header file with a bunch of useful HMM functions

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
		1.00  Initial release containing basic discrete and continuous hmm functions

 CITATION

 If you use this HMM Toolkit in a publication, please reference:
 Voshol, G.P. (2019). HMMTK: A simple HMM Toolkit (Version 1.0) [Software]. 
 Available from https://github.com/gerbenvoshol/Hidden-Markov-Model-Toolkit
