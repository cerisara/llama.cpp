ma proposition d'edition ressemble beaucoup à ROME/MEMIT, mais
est different sur plusieurs points:
- je n'ai pas besoin de "chercher" l'endroit ou se trouve une information, car je ne veux pas supprimer d'information mais en rajouter une nouvelle
- je n'ai pas besoin de faire l'hypothese qu'une info est localisee, mais par contre je fais l'hypothese que je peux
ajouter une nouvelle information de maniere localisee, ce qui est une hyp moins forte: c'est OK pour moi si une info
existante n'est pas localisee. L'info que j'ajoute est localisee, mais apres avoir ajoute plusieurs infos, il faudra
ensuite faire une low-rank compression des matrices, ou, en utilisant la methode de Yaya, faire une low-rank compression de plusieurs layers en meme temps ce qui peut permettre de transformer une info localisee en info non localisee !
- je n'ai pas besoin de faire de passe forward jusqu'en haut, je peux m'arreter a la layer a laquelle j'ajoute l'info
- on peut me reprocher de ne pas supprimer des infos obsoletes, mais des papiers ont montre que ROME/MEMIT et les autres
methodes d'edition ne suppriment pas vraiment une info.

Mon algo:
- le prompt est composé de 2 parties: P1 = nouvelle info + P2 = question
je suppose que lorsque j'ajoute P1+P2, le LLM trouve la bonne reponse mais que si je ne donne que P2, alors il donne une mauvaise reponse
- je suppose que l'information se trouve dans les activations du computation graph juste au timestep T1 = la fin de P1
- je propose alors de regarder toutes les activations des MLP au temps T1 = A1, ..., AL
- j'essaye ensuite d'ajouter un vecteur (K=Ai,V=Ai+1) au MLP i, et je teste la reponse du LLM sur seulement P2
- je teste cela pour les 30 a 70 MLP possibles, et je garde le meilleur

afaik, c'est la methode de model edition la moins couteuse !

