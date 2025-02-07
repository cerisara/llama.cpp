#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>

int main(int argc, char **argv) {
    char filename0[256];
    snprintf(filename0, sizeof(filename0), "./bin_tensors/%s", argv[1]);
    printf("reading %s\n", filename0);
    FILE *f = fopen(filename0, "rb");
	int vecdim=-1;
    int n_tok0=-1;
	fread(&vecdim, sizeof(int), 1, f);
	fread(&n_tok0, sizeof(int), 1, f);
    float *v0 = new float[100*10000*100]; // max 100 layers 10000 vecdim 100 tokens
    int n0=fread(v0,sizeof(float),100*10000*100,f);
    fclose(f);

    char filename1[256];
    snprintf(filename1, sizeof(filename1), "./bin_tensors/%s", argv[2]);
    printf("reading %s\n", filename1);
    f = fopen(filename1, "rb");
    int n_tok1=-1;
	fread(&vecdim, sizeof(int), 1, f);
    fread(&n_tok1, sizeof(int), 1, f);
    float* v1 = new float[100*10000*100]; // max 100 layers 10000 vecdim 100 tokens
    int n1=fread(v1,sizeof(float),100*10000*100,f);
    fclose(f);
    int n_tok = std::min(n_tok0, n_tok1);
    int n = std::min(n0, n1);

    printf("comparison between %s and %s\n", argv[1], argv[2]);
    printf("nread %d %d\n",n0,n1);
    printf("vecdim %d\n", vecdim);
    printf("n_tok %d\n", n_tok);

    int nlayers = n/(vecdim*n_tok);
    for (int l=0;l<nlayers;l++) {
        for (int t=0;t<n_tok;t++) {
            float d=0.;
            float u=0.;
            float v=0.;
            double prod_sum=0.;
            double square_sum0=0.;
            double square_sum1=0.;
            for (int i=0;i<vecdim;i++) {
                float d0=v0[l*n_tok0*vecdim+vecdim*t+i];
                float d1=v1[l*n_tok1*vecdim+vecdim*t+i];
                float dd=d0-d1;
                d+=dd*dd;
                u+=d0*d0;
                v+=d1*d1;
                prod_sum+=d0*d1;
            }
            double cosine_sim = prod_sum / (sqrt(u)*sqrt(v));
            printf("layer: %d token: %d normDIFF: %f normICL: %f normERR: %f Cosine sim: %f\n", l, t, d, u, v, cosine_sim);
        }
    }
    delete[] v0;
    delete[] v1;
    return 0;
}
