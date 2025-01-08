#include <stdio.h>

void main(int argc, char **argv) {
    FILE *f = fopen(argv[1], "rb");
	int vecdim=-1;
	fread(&vecdim, sizeof(int), 1, f);
    float v[100*5000]; // max 100 layers 5000 vecdim
    int n=fread(v,sizeof(float),100*5000,f);
    fclose(f);

    printf("showing %s\n", argv[1]);
    printf("nread %d\n",n);
    for (int i=0;i<100;i++) printf("%f ",v[i]);
    printf("\n");

    int nlayers = n/vecdim;
    for (int l=0;l<nlayers;l++) {
        float d=0.;
        for (int i=0;i<vecdim;i++) {
            float dd=v[l*(vecdim+1)+i];
            d+=dd*dd;
        }
        printf("%d %f\n",l,d);
    }
}
