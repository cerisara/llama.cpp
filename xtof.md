This code is a modification of the vanilla llama.cpp that adds the following 2 functionalities:

1- save all activations (set env var DETSAVE=1)
2- edit some activations during the forward pass (set env var DETADD=path to filename)

For the latter, the filename is a text file that should be created before calling llama.cpp and for which
each line contains 3 int and 1 float for respectively: layer token dim bias2add
For the former, the activations are saved in the text file ./acts.bin (see format in all.py)

----------

Process:
- chargement du model dans une structure "model"
- creation du buffer pour stocker le futur graphe
- a chaque token, creation du compute graph vide
- ensuite, allocation du graphe dans le buffer

---------------

## Method

qLoRA requires backprop through the graph, which is costly for large LLMs.
LadderSideNets avoid this backprop, but only use the LLM as a feature computer,
and train another model from these features. I propose to rather train a side
model that reinjects its output into the main LLM, in a similar way as qLoRA, but
its training is done with a local distillation loss in a teacher-student fashion,
which avoids the need to backprop in the LLM.
The distillation process is based on a teacher that is the LLM augmented with in-context learning,
where some new information to learn is written in the prompt so that the LLM answers
correctly to the input question about this piece of information (that we assume the initial LLM
does not know). The activations at the output of one layer in the LLM are then considered
as the teacher activations that we want to obtain again but with just the input question, without
in-context learning. We then train an activation bias that is added to the activations at the positions
of the input question only and that transforms the student activations
of this layer (without ICL) into the teacher one (with ICL). 
This bias is obtained through a small transformer that is distilled from the MSE loss on both these activations.

---------------

est-ce que c'est une bonne idee de prendre LORA comme exemple ?
est-ce que LORA ne transforme/integre pas les poids du modele avec LORA au moment de l'init ?

Ne pas confondre:
- le "model" qui est cree au moment de l'init, avant le warmup; La fct qui applique LORA est appelee avant le warmup
- le compute graph qui est recree a chaque token

Pour le moment, je modifie le compute graph, alors que LORA modifie le modele !
C'est peut-etre plus simple de reprendre ce que je faisais au debut == modifier lors de la propagation des inputs dans le graphe le calcul dans les noeuds "lout". Mais ce calcul est different en cuda que sur CPU, et il me faudrait gerer le transfert des tensors du CPU vers le GPU.

Donc c'est sans doute mieux de modifier le compute graph, mais il faut que je libere la RAM

