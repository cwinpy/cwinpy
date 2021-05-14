#include "filter_core.h"

double filter_core(double x, int nrecurs, double *recursCoef, int ndirect, double *directCoef, int nhistory, double *history){
    /*
    Infinite Impulse response filter code taken from that for the
    XLALIIRFilterREAL8 in LAL - see https://lscsoft.docs.ligo.org/lalsuite/lal/group___i_i_r_filter__c.html
    */
    
    int j;           /* Index for filter coefficients */
    int jmax;        /* Number of filter coefficients */
    double *coef;    /* Values of filter coefficients */
    double *hist;    /* Values of filter history */
    double w;        /* Auxiliary datum */
    double y;        /* Output datum */

    /* Compute the auxiliary datum */
    jmax = nrecurs;
    coef = recursCoef+1;
    hist = history;
    w = x;
    for (j = 1; j < jmax; j++){
        w += (*(coef++)) * (*(hist++));
    }
    hist -= (jmax - 1);

    /* Compute the filter output */
    jmax = ndirect;
    coef = directCoef;
    y = (*(coef++)) * w;
    for (j = 1; j < jmax; j++){
        y += (*(coef++)) * (*(hist++));
    }
    hist -= (jmax - 1);

    /* Update the filter history */
    jmax = nhistory - 1;
    hist += jmax;
    for (j = jmax; j > 0; j--, hist--){
        *hist = hist[-1];
    }
    *hist = w;

    return y;
}