
Il y a 2 programmes:

- modif de llama-cli (examples/main/main.cpp): SAVEACT=1 enregistre "acts.bin"=FFN activations
- new code (examples/detson/detgguf.cpp lance avec ./build/bin/detgguf) qui ajoute une dim aux FFN

gradient-free:
- forward gradient learning
- node perturbation = add noise, check the change in final loss = still requires the final loss
    - see EFFECTIVE LEARNING WITH NODE PERTURBATION IN DEEP NEURAL NETWORKS
- weight propagation

---------------------

- pourquoi eval_cb est called 2 fois ? ==> il peut etre appele plusieurs fois, pour savoir si un noeud a besoin des data

- XP1: 2 passes forward avec info + Q et seulement Q: pas de diff particuliere des activations des MLP entre les 2.
  Il faudra donc tester toutes les layers une par une.

- TODO: modifier un gguf de LLM pour lui ajouter des vecteurs (K, gate, V)
  il faut bien comprendre le graphe pour cela


ma proposition d'edition ressemble beaucoup à ROME/MEMIT, mais
est different sur plusieurs points:
- je n'ai pas besoin de "chercher" l'endroit ou se trouve une information dans toute la matrice, car je ne veux pas supprimer d'information mais en rajouter une nouvelle, mais je cherche le concept seulement dans la derniere trame
- apres avoir ajoute plusieurs infos, il faudra
ensuite faire une low-rank compression des matrices, ou, en utilisant la methode de Yaya, faire une low-rank compression de plusieurs layers en meme temps ce qui peut permettre de transformer une info localisee en info non localisee !
- je n'ai pas besoin de faire de passe forward jusqu'en haut, je peux m'arreter a la layer a laquelle j'ajoute l'info
- on peut me reprocher de ne pas supprimer des infos obsoletes, mais des papiers ont montre que ROME/MEMIT et les autres
methodes d'edition ne suppriment pas vraiment une info.

Mon algo:
- le prompt est composé de 2 parties: P1 = nouvelle info + P2 = question
je suppose que lorsque j'ajoute P1+P2, le LLM trouve la bonne reponse mais que si je ne donne que P2, alors il donne une mauvaise reponse
- je suppose que le fact que je veux ajouter est modélisé par 2 vecteurs: (K,V) et que, au dernier timestep (à la fin de P2), un MLP dans une layer prend en entrée K==X le concept représentant la questions, et output V==Y le concept de la bonne réponse.
- pour trouver ces concepts, je fais 2 passes forward: avec P1+P2 et avec P2: le concept de la question doit apparaître dans les 2 passes avec des concepts de réponse différents, je cherche donc les X de chaque layer qui sont les plus proches (au dernier timestep) et qui ont un Y différent
- j'essaye ensuite d'ajouter un vecteur (K=Ai,V=Ai+1) au MLP i, et je teste la reponse du LLM sur seulement P2
- je teste cela pour les 30 a 70 MLP possibles, et je garde le meilleur

afaik, c'est la methode de model edition la moins couteuse !

