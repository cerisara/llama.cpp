#include <stdio.h>

void main(int argc, char **argv) {
    FILE *f = fopen("acts.bin","rb");
    float v[100*5000]; // max 100 layers 5000 vecdim
    int n=fread(v,sizeof(float),100*5000,f);
    fclose(f);

    printf("nread %d\n",n);
    for (int i=0;i<100;i++) printf("%f ",v[i]);
    printf("\n");

    // TODO: recuperer le vecdim pour etre general
    int nlayers = n/3584;
    for (int l=0;l<nlayers;l++) {
        float d=0.;
        for (int i=0;i<3584;i++) {
            float dd=v[l*3584+i];
            d+=dd*dd;
        }
        printf("%d %f\n",l,d);
    }
}
