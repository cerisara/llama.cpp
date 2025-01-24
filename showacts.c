#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void main(int argc, char **argv) {
    char* filename;
    filename = malloc(strlen(argv[1])+15);
    strcpy(filename, "./bin_tensors/");
    strcat(filename, argv[1]);
    FILE *f = fopen(filename, "rb");
	int vecdim=-1;
    int n_tok=-1;
	fread(&vecdim, sizeof(int), 1, f);
	fread(&n_tok, sizeof(int), 1, f);
    printf("vecdim %d\n",vecdim);
    printf("ntok %d\n", n_tok);
    float v[100*10000]; // max 100 layers 10000 vecdim*n_tok
    int n=fread(v,sizeof(float),100*10000,f);
    fclose(f);

    printf("showing %s\n", argv[1]);
    printf("nread %d\n",n);
    for (int i=0;i<100;i++) printf("%f ",v[10*n_tok*vecdim+i+vecdim+vecdim-260]);
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
}
