#include "mex.h"
#include <vector>
#include <math.h>
#include<algorithm>
using namespace std;


void  mexFunction(int nlhs, mxArray *out[], int nrhs, const mxArray *input[]) {

    if(mxGetClassID(input[0])!=mxSINGLE_CLASS || mxGetClassID(input[1])!=mxSINGLE_CLASS) mexErrMsgTxt("first input must be single");
    
    float* feat_map = (float*)mxGetPr( input[0] ); //nsample * nfeat
    int nfeat = (int) mxGetM(input[0]); 
    int nsample = (int) mxGetN(input[0]);
    
    float* prop_inds = (float*)mxGetPr( input[1] );
    int nprop = (int) mxGetM(input[1]); 
    //mexPrintf("feat_map[74, 12]: %f \n", feat_map[73 + 11*nsample]);
    
    
    int i1,i2,i3,i4,x0,x1,x2,x3,x4;
    //out[0] = mxCreateNumericMatrix(nprop,nfeat,mxSINGLE_CLASS,mxREAL);
    //float *ing_feat = (float*) mxGetData(out[0]); //output
    out[0] = mxCreateNumericMatrix(nfeat,nprop,mxSINGLE_CLASS,mxREAL);
    float *ing_feat = (float*) mxGetData(out[0]); //output
    for(int i=0; i<nprop; i++) {      
      i1 = (int)prop_inds[ i + 0*nprop ]-1;
      i2 = (int)prop_inds[ i + 1*nprop ]-1;
      i3 = (int)prop_inds[ i + 2*nprop ]-1;
      i4 = (int)prop_inds[ i + 3*nprop ]-1;
      //i1 = (int)prop_inds[ 4*i + 0 ]-1;
      //i2 = (int)prop_inds[ 4*i + 1 ]-1;
      //i3 = (int)prop_inds[ 4*i + 2 ]-1;
      //i4 = (int)prop_inds[ 4*i + 3 ]-1;
      
    
      x0=i*nfeat; x1=i1*nfeat; x2=i2*nfeat; x3=i3*nfeat; x4=i4*nfeat;
      for(int j=0; j<nfeat; j++)
	ing_feat[ x0 + j ] = feat_map[x1 + j] - feat_map[x2 + j] - feat_map[x3 + j] + feat_map[x4 + j];
      
      //for(int j=0; j<nfeat; j++)
	//ing_feat[ i + j*nprop ] = feat_map[i1 + j*nsample] - feat_map[i2 + j*nsample] - feat_map[i3 + j*nsample] + feat_map[i4 + j*nsample];
      
    }
    
}
