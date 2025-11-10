1. activer environnemment : source machin chouette
2. Faire : pip install -r requirements.txt
2. Faire : python3 src/swarm_rescue/launcher.py --config config/my_eval_plan.yml
On se concentre d'abord sur la map la plus simple

Jacques et Come : on s'est concentré sur le controle du drone, il n'est pas autonome
On a donné la trajectoire à suivre en vert.
Le drone affiche aussi la map LIDAR

Pour rendre le drone autonome, il faut implémenter une fonction qui définit le point objectif à atteindre et une fonction qui prend le point goal à atteindre et output la trajectoire à suivre.
