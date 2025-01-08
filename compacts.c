#include <stdio.h>

void main(int argc, char **argv) {
    FILE *f = fopen(argv[1], "rb");
	int vecdim=-1;
	fread(&vecdim, sizeof(int), 1, f);
    float v0[100*5000]; // max 100 layers 5000 vecdim
    int n0=fread(v0,sizeof(float),100*5000,f);
    fclose(f);

    f = fopen(argv[2], "rb");
	fread(&vecdim, sizeof(int), 1, f);
    float v1[100*5000]; // max 100 layers 5000 vecdim
    int n1=fread(v1,sizeof(float),100*5000,f);
    fclose(f);

    printf("comparison between %s and %s\n", argv[1], argv[2]);
    printf("nread %d %d\n",n0,n1);

    int nlayers = n0/vecdim;
    for (int l=0;l<nlayers;l++) {
        float d=0.;
        float u=0.;
        float v=0.;
        for (int i=0;i<vecdim;i++) {
            float dd=v1[l*(vecdim+1)+i]-v0[l*(vecdim+1)+i];
            d+=dd*dd;
            dd=v0[l*(vecdim+1)+i];
            u+=dd*dd;
            dd=v1[l*(vecdim+1)+i];
            v+=dd*dd;
        }
        printf("layer: %d normDIFF: %f normICL: %f normERR: %f\n",l,d,u,v);
    }
}
