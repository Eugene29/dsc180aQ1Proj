#!/bin/bash
# #!/geom/bin/python

# Conv: GCN, GIN, GAT, GraphGPS
# args: Dataset | Conv | Virtual Node
# accelerate launch --num_processes=3 experiment.py Cora GAT False
# accelerate launch experiment.py Peptides-func GCN False
# accelerate launch experiment.py Peptides-func GIN False 
# accelerate launch experiment.py Peptides-func GAT False 
# accelerate launch experiment.py Peptides-func GraphGPS False ## .665

accelerate launch experiment.py Cora GIN False
accelerate launch experiment.py Enzymes GCN False
accelerate launch experiment.py IMDB GIN False
accelerate launch experiment.py Peptides-func GraphGPS False

# accelerate launch experiment.py Enzymes GAT False
# accelerate launch experiment.py IMDB GAT False
# accelerate launch experiment.py Peptides-func GraphGPS False