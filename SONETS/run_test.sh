
#for seed in {0..36}; do
#./run_secorder 1250 0.2 0.0 0.0 0.0 0 $seed
#./run_secorder 1250 0.2 0.1 0.0 0.0 0 $seed
#./run_secorder 1250 0.2 0.2 0.0 0.0 0 $seed
#./run_secorder 1250 0.2 0.3 0.0 0.0 0 $seed
#./run_secorder 1250 0.2 0.4 0.0 0.0 0 $seed
#./run_secorder 1250 0.2 0.5 0.0 0.0 0 $seed
#./run_secorder 1250 0.2 0.6 0.0 0.0 0 $seed
#./run_secorder 1500 0.2 0.7 0.0 0.0 0 $seed
#./run_secorder 1500 0.2 0.8 0.0 0.0 0 $seed
#./run_secorder 1250 0.24 1.2 0.6 0.6 1 $seed
#done

for seed in {0..36}; do
./run_secorder 5000 0.06 1.2 0.6 0.6 1 $seed
done