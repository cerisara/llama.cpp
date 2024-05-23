
Process:
- chargement du model dans une structure "model"
- creation du buffer pour stocker le futur graphe
- a chaque token, creation du compute graph vide
- ensuite, allocation du graphe dans le buffer

---------------

est-ce que c'est une bonne idee de prendre LORA comme exemple ?
est-ce que LORA ne transforme/integre pas les poids du modele avec LORA au moment de l'init ?

Ne pas confondre:
- le "model" qui est cree au moment de l'init, avant le warmup; La fct qui applique LORA est appelee avant le warmup
- le compute graph qui est recree a chaque token

Pour le moment, je modifie le compute graph, alors que LORA modifie le modele !
C'est peut-etre plus simple de reprendre ce que je faisais au debut == modifier lors de la propagation des inputs dans le graphe le calcul dans les noeuds "lout". Mais ce calcul est different en cuda que sur CPU, et il me faudrait gerer le transfert des tensors du CPU vers le GPU.

Donc c'est sans doute mieux de modifier le compute graph, mais il faut que je libere la RAM

