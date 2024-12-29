#include <stdio.h>

void main(int argc, char **argv) {
    FILE *f = fopen("acts.bin.gld","rb");
	int vecdim=-1;
	fread(&vecdim, sizeof(int), 1, f);
    float v0[100*5000]; // max 100 layers 5000 vecdim
    int n0=fread(v0,sizeof(float),100*5000,f);
    fclose(f);

    f = fopen("acts.bin.rec","rb");
	fread(&vecdim, sizeof(int), 1, f);
    float v1[100*5000]; // max 100 layers 5000 vecdim
    int n1=fread(v1,sizeof(float),100*5000,f);
    fclose(f);

    printf("nread %d %d\n",n0,n1);

    int nlayers = n0/vecdim;
    for (int l=0;l<nlayers;l++) {
        float d=0.;
        float u=0.;
        float v=0.;
        for (int i=0;i<vecdim;i++) {
            float dd=v1[l*vecdim+i]-v0[l*vecdim+i];
            d+=dd*dd;
            dd=v0[l*vecdim+i];
            u+=dd*dd;
            dd=v1[l*vecdim+i];
            v+=dd*dd;
        }
        printf("layer: %d normDIFF: %f normICL: %f normERR: %f\n",l,d,u,v);
    }
}
