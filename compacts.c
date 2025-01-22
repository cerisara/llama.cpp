#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

void main(int argc, char **argv) {
    char* filename;
    filename = malloc(strlen(argv[1])+15);
    strcpy(filename, "./bin_tensors/");
    strcat(filename, argv[1]);
    printf("reading %s\n", filename);
    FILE *f = fopen(filename, "rb");
	int vecdim=-1;
    int n_tok=-1;
	fread(&vecdim, sizeof(int), 1, f);
	fread(&n_tok, sizeof(int), 1, f);
    float v0[100*10000]; // max 100 layers 10000 vecdim*n_tok
    int n0=fread(v0,sizeof(float),100*10000,f);
    fclose(f);

    filename = malloc(strlen(argv[2])+15);
    strcpy(filename, "./bin_tensors/");
    strcat(filename, argv[2]);
    printf("reading %s\n", filename);
    f = fopen(filename, "rb");
	fread(&vecdim, sizeof(int), 1, f);
    fread(&n_tok, sizeof(int), 1, f);
    float v1[100*10000]; // max 100 layers 10000 vecdim*n_tok
    int n1=fread(v1,sizeof(float),100*10000,f);
    fclose(f);

    printf("comparison between %s and %s\n", argv[1], argv[2]);
    printf("nread %d %d\n",n0,n1);
    printf("vecdim %d\n", vecdim);
    printf("n_tok %d\n", n_tok);

    int nlayers = n0/(vecdim*n_tok);
    for (int l=0;l<nlayers;l++) {
        for (int t=0;t<n_tok;t++) {
            float d=0.;
            float u=0.;
            float v=0.;
            double prod_sum=0.;
            double square_sum0=0.;
            double square_sum1=0.;
            for (int i=0;i<vecdim;i++) {
                float d0=v0[l*n_tok*vecdim+vecdim*t+i];
                float d1=v1[l*n_tok*vecdim+vecdim*t+i];
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
}
