#include "mex.h"
#include <vector>
#include <math.h>
#include<algorithm>
using namespace std;
int clamp( int v, int a, int b ) { return v<a?a:v>b?b:v; }

// bounding box data structures and routines
typedef struct { int c, r, w, h; float s; } Box;
typedef vector<Box> Boxes;
bool boxesCompare( const Box &a, const Box &b ) { return a.s<b.s; }
float boxesOverlap( Box &a, Box &b );
void boxesNms( Boxes &boxes, float thr, int maxBoxes, float eta );



float boxesOverlap( Box &a, Box &b ) {
  float areai, areaj, areaij;
  int r0, r1, c0, c1, r1i, c1i, r1j, c1j;
  r1i=a.r+a.h; c1i=a.c+a.w; if( a.r>=r1i || a.c>=c1i ) return 0;
  r1j=b.r+b.h; c1j=b.c+b.w; if( a.r>=r1j || a.c>=c1j ) return 0;
  areai = (float) a.w*a.h; r0=max(a.r,b.r); r1=min(r1i,r1j);
  areaj = (float) b.w*b.h; c0=max(a.c,b.c); c1=min(c1i,c1j);
  areaij = (float) max(0,r1-r0)*max(0,c1-c0);
  return areaij / (areai + areaj - areaij);
}




void boxesNms( Boxes &boxes, float thr, int maxBoxes, float eta )
{
  sort(boxes.rbegin(),boxes.rend(),boxesCompare);
  if( thr>.99 ) return; const int nBin=10000;
  const float step=1/thr, lstep=log(step);
  vector<Boxes> kept; kept.resize(nBin+1);
  int i=0, j, k, n=(int) boxes.size(), m=0, b, d=1;
  while( i<n && m<maxBoxes ) {
    b = boxes[i].w*boxes[i].h; bool keep=1;
    b = clamp(int(ceil(log(float(b))/lstep)),d,nBin-d);
    for( j=b-d; j<=b+d; j++ )
      for( k=0; k<kept[j].size(); k++ ) if( keep )
        keep = boxesOverlap( boxes[i], kept[j][k] ) <= thr;
    if(keep) { kept[b].push_back(boxes[i]); m++; } i++;
    if(keep && eta<1 && thr>.5) { thr*=eta; d=ceil(log(1/thr)/lstep); }
  }
  boxes.resize(m); i=0;
  for( j=0; j<nBin; j++ )
    for( k=0; k<kept[j].size(); k++ )
      boxes[i++]=kept[j][k];
  sort(boxes.rbegin(),boxes.rend(),boxesCompare);
  
}


void  mexFunction(int nlhs, mxArray *out[], int nrhs, const mxArray *input[]) {
    
    float thr=0.5, eta=1;
    int maxBoxes=100000;
    if(mxGetClassID(input[0])!=mxSINGLE_CLASS) mexErrMsgTxt("first input must be single");
    if (nrhs==4) eta = (float) mxGetScalar(input[3]);
    if (nrhs>=3) maxBoxes = (int) mxGetScalar(input[2]);
    if (nrhs<2) mexErrMsgTxt("Usage: nms_c(boxes, thre, max_nbox=Inf, eta=1)");
    
    thr = (float) mxGetScalar(input[1]);    
    float* boxes_array = (float*)mxGetPr( input[0] );
    int nbox = (int) mxGetM(input[0]);  //number of input boxes
    //mexPrintf("nbox: %d, thr: %f \n", nbox, thr);
    
    int x2,y2;
    Boxes boxes;
    boxes.resize(0);
    for(int i=0; i<nbox; i++) {
      Box b;
      b.c = (int)boxes_array[ i + 0*nbox ]-1;
      b.r = (int)boxes_array[ i + 1*nbox ]-1;
      x2  = (int) boxes_array[ i + 2*nbox ]-1;
      y2  = (int) boxes_array[ i + 3*nbox ]-1;
      b.w = (int) x2 - b.c + 1;
      b.h = (int) y2 - b.r + 1;
      b.s = (float) boxes_array[ i + 4*nbox ];
      boxes.push_back(b);
    }
        
    boxesNms(boxes, thr, maxBoxes, eta);
        
    //output
    int n = (int) boxes.size();
    out[0] = mxCreateNumericMatrix(n,5,mxSINGLE_CLASS,mxREAL);
    float *bbs = (float*) mxGetData(out[0]);
    for(int i=0; i<n; i++) {
      bbs[ i + 0*n ] = (float) boxes[i].c+1;
      bbs[ i + 1*n ] = (float) boxes[i].r+1;
      bbs[ i + 2*n ] = (float) (boxes[i].c+boxes[i].w);
      bbs[ i + 3*n ] = (float) (boxes[i].r+boxes[i].h);
      bbs[ i + 4*n ] = boxes[i].s;
    }
    
    
}