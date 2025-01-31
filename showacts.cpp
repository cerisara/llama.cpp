#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    char filename[256];
    snprintf(filename, sizeof(filename), "./bin_tensors/%s", argv[1]);
    FILE *f = fopen(filename, "rb");
	int vecdim=-1;
    int n_tok=-1;
	fread(&vecdim, sizeof(int), 1, f);
	fread(&n_tok, sizeof(int), 1, f);
    printf("vecdim %d\n",vecdim);
    printf("ntok %d\n", n_tok);
    float* v = new float[100*10000*100]; // max 100 layers 10000 vecdim 100 tokens
    int n=fread(v,sizeof(float),100*10000*100,f);
    fclose(f);

    printf("showing %s\n", argv[1]);
    printf("nread %d\n",n);
    for (int i=-1;i<n_tok+1;i++) {
        for (int j=-1;j<n_tok+1;j++) {
            printf("%f ",v[0*n_tok*vecdim+i+j*vecdim-256]);
        }
        printf("\n");
    }
    printf("\n");

    int nlayers = n/(vecdim*n_tok);
    for (int l=0;l<nlayers;l++) {
        for (int t=0;t<n_tok;t++) {
            float d=.0;
            for (int i=0;i<vecdim;i++) {
                float dd=v[l*n_tok*vecdim+vecdim*t+i];
                d+=dd*dd;
            }
            printf("%d %d %f\n",l, t, d);
        }
    }
    
    delete[] v;
    return 0;
}
