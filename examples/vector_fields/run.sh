#!/bin/bash -l

brainbuilder atlases -n 6,5,4,3,2,1 -t 700,525,190,353,149,165 -d 10 -o atlas column -a 1000
python plot_vector_fields.py
