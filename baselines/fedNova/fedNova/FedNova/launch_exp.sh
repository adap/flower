# run FedAvg with local momentum
# h0,h1,..,h15 are the hosts names in our private cluster
# you need to replace them either by IP address or your own host names
pdsh -R ssh -w h[0-15] "python3.5 /users/jianyuw1/FedNova/train_LocalSGD.py --pattern constant \
                        --lr 0.02 --bs 32 --localE 2 --alpha 0.1 --mu 0 --momentum 0.9 \
                        --save -p --name FedAvg_momen_baseline --optimizer fedavg --model VGG \
                        --rank %n --size 16 --backend nccl --initmethod tcp://h0:22000 \
                        --rounds 100 --seed 1 --NIID --print_freq 50"

# run FedNova with local momentum
# h0,h1,..,h15 are the hosts names in our private cluster
# you need to replace them either by IP address or your own host names
pdsh -R ssh -w h[0-15] "python3.5 /users/jianyuw1/FedNova/train_LocalSGD.py --pattern constant \
                        --lr 0.02 --bs 32 --localE 2 --alpha 0.1 --mu 0 --momentum 0.9 \
                        --save -p --name FedNova_momen_baseline --optimizer fednova --model VGG \
                        --rank %n --size 16 --backend nccl --initmethod tcp://h0:22000 \
                        --rounds 100 --seed 3 --NIID --print_freq 50"