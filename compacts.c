#include <stdio.h>

void main(int argc, char **argv) {
    FILE *f = fopen("acts.bin.icl","rb");
    float v0[100*5000]; // max 100 layers 5000 vecdim
    int n0=fread(v0,sizeof(float),100*5000,f);
    fclose(f);

    f = fopen("acts.bin.err","rb");
    float v1[100*5000]; // max 100 layers 5000 vecdim
    int n1=fread(v1,sizeof(float),100*5000,f);
    fclose(f);

    printf("nread %d %d\n",n0,n1);

    // TODO: recuperer le vecdim pour etre general
    int nlayers = n0/3584;
    for (int l=0;l<nlayers;l++) {
        float d=0.;
        float u=0.;
        float v=0.;
        for (int i=0;i<3584;i++) {
            float dd=v1[l*3584+i]-v0[l*3584+i];
            d+=dd*dd;
            dd=v0[l*3584+i];
            u+=dd*dd;
            dd=v1[l*3584+i];
            v+=dd*dd;
        }
        printf("layer: %d normDIFF: %f normICL: %f normERR: %f\n",l,d,u,v);
    }
}
