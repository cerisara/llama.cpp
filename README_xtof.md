
       KQVout     l_out
            \     /
             ffn_up
               |
             norm
               |
       W    ffn_norm    W
        \0   /    \    /0
       ffn_gate   ffn_up
          |          │
       ffn_silu      │
               \     │
                ffn_gate_par   W
                     |         │
                   ffn_out─────┘0



Il y a 2 main programmes (cf. xtof.sh pour les compiler):

- modif de llama-cli (examples/main/main.cpp): SAVEACT=1 enregistre "acts.bin"=FFN activations
- new code (examples/detson/detgguf.cpp lance avec ./build/bin/detgguf) qui ajoute une dim aux FFN

il y a 2 programmes secondaires, mais qui ne sont pas importants:
- showacts.c: affiche la norme des activations en sortie des MLP
- compacts.c: affiche la norme du delta entre les activations gold et erronees

---------------------

PRINCIPE: "knowledge insertion in LLMs"

Given a question Q that has a single-word answer, and an ICL prompt P that contains a
piece of knowledge related to question Q, so that when the LLM is asked Q, it answers
incorrectly, while when the LLM answer P+Q, it answers correctly.
With P+Q, the latent activations at the output of MLPs contain the correct knowledge,
while with only Q, they do not.
Assuming knowledge is stored in the MLPs, we make the hypothesis that this new knowledge,
which has been injected in the prompt, could instead be inserted as an additional set of vectors
into one (or several ?) MLP, so that the LLM answers correctly when only Q is prompted.

In Qwen, the MLPs are gated MLPs with 3 matrices:
- The "up-projection" matrix, which contains N "key vectors" that are compared to the input "query" and outputs the scalar similarity between each key vector and the input.
- The gate matrix, which also contains N key vectors but outputs a scalar that will inhibate or not each corresponding vector in the up-projection matrix
- The "down-projection" matrix, which contains N "value vectors" that are combined to form the output value, according to both previously computed similarities and gates

Let us focus on one MLP: assuming Xg is its input "query" with P+Q and Xr with only Q, and similarly Yg and Ys its output values.
We want the MLP to output Yg when seeing Xr.
So we append one row to both the up and gate matrices that contains Xr, so that this new vector perfectly matches Xr and
is activated when Xr is presented. And we append one column to the down matrix with Yg, so that this target value is
outputed when Xr is presented.
This new dimension should "dominate" the other vectors in the MLP when Q is prompted to the LLM.
We further hypothetise that, when such an information is added around the middle of the LLM layers, then it encodes the
abstract concept related to Q, and not the surface form of Q, hence enabling generalization to paraphrases and
further reasoning with this piece of information.

TODO:
- XP by inserting knowledge in a single MLP, test with each possible layer
- if this does not work, test inserting in multiple or even all layers, but this may require starting from the bottom
  and recomputing the Xr after each layer insertion.


---------------------

SOTA gradient-free:
- forward gradient learning
- node perturbation = add noise, check the change in final loss = still requires the final loss
    - see EFFECTIVE LEARNING WITH NODE PERTURBATION IN DEEP NEURAL NETWORKS
- weight propagation

BASELINE with supervised finetuning:
- Injecting New Knowledge into Large Language Models via Supervised Fine-Tuning
    - possible d'injecter knowledge with SFT en dupliquant+reformulant 10x les faits
    - RAG reste toujours meilleur
- Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs
    - memes auteurs et conclusions
- Structure-aware Domain Knowledge Injection for Large Language Models
    - continued pretraining
- Learning to Edit: Aligning LLMs with Knowledge Editing

---------------------

PB: l'info du ICL n'est peut-etre pas utilisee immediatement apres la question, mais le LLM
peut d'abord raisonner puis aller chercher cette info plus tard.
==> il faut des questions dont la reponse tient sur un seul mot !

---------------------

- pourquoi eval_cb est called 2 fois ? ==> il peut etre appele plusieurs fois, pour savoir si un noeud a besoin des data

- XP1: 2 passes forward avec info + Q et seulement Q: pas de diff particuliere des activations des MLP entre les 2.
  Il faudra donc tester toutes les layers une par une.

- TODO: modifier un gguf de LLM pour lui ajouter des vecteurs (K, gate, V)
  il faut bien comprendre le graphe pour cela


ma proposition d'edition ressemble beaucoup à ROME/MEMIT, mais est different sur plusieurs points:
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

