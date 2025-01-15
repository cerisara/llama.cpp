#include <stdio.h>

int main(int argc, char **argv) {
    FILE *f = fopen("acts.bin.gld", "rb");
	int vec_dim = -1;
	fread(&vec_dim, sizeof(int), 1, f);
    float v0[100*5000]; // max 100 layers 5000 vecdim
    int n0 = fread(v0, sizeof(float), 100*5000, f);
    fclose(f);

    int nlayers = n0/vec_dim;
    float best_norm_diff = 1000000000;
    int best_layer = -1;

    for (int i=10; i<13; i++) {
        char acts_file[20];
        sprintf(acts_file, "acts.bin.rec_%d", i);
        f = fopen(acts_file, "rb");
        fread(&vec_dim, sizeof(int), 1, f);
        float v1[100*5000]; // max 100 layers 5000 vecdim
        int n1 = fread(v1,sizeof(float),100*5000,f);
        fclose(f);

        printf("Evaluating layer%d\n", i);
        printf("nread %d\n",n1);

        float norm_diff=0.;
        for (int j=0; j<vec_dim; j++) {
            float diff = v1[nlayers*(vec_dim+1)+j] - v0[nlayers*(vec_dim+1)+j];
            norm_diff += diff * diff;
        }

        if (norm_diff < best_norm_diff) {
            best_norm_diff = norm_diff;
            best_layer = i;
        }
    }

    printf("Best model: %d\n", best_layer);
    return 0;
}