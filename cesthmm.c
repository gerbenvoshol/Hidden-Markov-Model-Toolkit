#include <stdio.h>
#include <math.h>

#define HMMTK_DEFINE
#include "hmmtk.h"

void Usage(char *name)
{
    printf("Usage error. \n");
    printf("Usage1: %s [-v] -N <num_states> -M <num_symbols> <file.seq>\n", 
    name);
    printf("Usage2: %s [-v] -S <seed> -N <num_states> -M <num_symbols> <file.seq>\n", 
    name);
    printf("Usage3: %s [-v] -I <mod.hmm> <file.seq>\n", 
    name);
    printf("  N - number of states\n");
    printf("  M - number of symbols\n");
    printf("  S - seed for random number genrator\n");
    printf("  I - mod.hmm is a file with the initial model parameters\n");
    printf("  file.data - file containing the obs. seqence\n");
    printf("  v - prints out number of iterations and log prob\n");
}

int main(int argc, char *argv[])
{
    CHMM hmm;
    int N = 27;
    int D = 4;
    int seed = 0;
    int c;
    int iflg=0, sflg=0, nflg=0, mflg=0, errflg =0, vflg=0;
    char    *hmminitfile;
    // int T = 100;

    extern char *optarg;
    extern int optind, opterr, optopt;

    while ((c= getopt(argc, argv, "vhI:S:N:M:")) != EOF) {
        switch (c) {
            case 'v': 
                vflg++; 
                break;
            case 'h': 
                Usage(argv[0]);
                exit(1);
                break;
            case 'S':
                /* set random number generator seed */
                if (sflg)
                        errflg++;
                else {
                        sflg++;
                        sscanf(optarg, "%d", &seed);
                }
                break;
            case 'N':  
                /* set random number generator seed */
                if (nflg) 
                        errflg++; 
                else { 
                        nflg++;  
                        sscanf(optarg, "%d", &N);
                } 
                break;   
            case 'M':  
                /* set random number generator seed */
                if (mflg) 
                        errflg++; 
                else { 
                        mflg++;  
                        sscanf(optarg, "%d", &D);
                } 
                break;   
            case 'I':  
                /* set random number generator seed */
                if (iflg) 
                    errflg++; 
                else { 
                    iflg++;  
                    hmminitfile = optarg;
                } 
                break;   
            case '?':
                errflg++;
        }
    }
    /* you can initialize the hmm model three ways:
           i) with a model stored in a file, which also sets 
          the number of states N and number of symbols M.
           ii) with a random model by just specifyin N and M
              on the command line.
           iii) with a specific random model by specifying N, M
              and seed on the command line. 
        */

    if (iflg) {
        /* model being read from a file */
        if (((sflg || nflg) || mflg)) {
            errflg++;
        }
    } else if ((!nflg) || (!mflg)) { 
        /* Model not being intialied from file */ 
        /* both N and M should be specified */
        errflg++; 
    }

    
    if ((argc - optind) != 1) errflg++; /* number or arguments not okay */

    if (errflg) {
        Usage(argv[0]);
        exit (1);
    }


    /* initialize the hmm model */
    if (iflg) { 
        chmm_load(hmminitfile, &hmm);
    } else if (sflg) {
        chmm_init(&hmm, N, "0ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz", D, "0123456789", seed);
    } else {
        seed = hmm_seed();
        chmm_init(&hmm, N, "0ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz", D, "0123456789", seed);
    }


    struct samples *p_samples;
    int iter;
    double logprobinit, logprobfinal;

    p_samples = chmm_load_samples(argv[optind]);

    // ///////
    // //
    // //
    // // remember to define DIM
    // //
    // //
    // ///////

    chmm_save("stdout", &hmm);

    chmm_baumwelch(&hmm, p_samples, &iter, &logprobinit, &logprobfinal, 100);

    chmm_save("stdout", &hmm);

    return 0;
}

