comm= ("poetry run python -m fedpara.main" ,
"poetry run python -m fedpara.main --config-name cifar100" ,
"poetry run python -m fedpara.main num_epochs=10 model.conv_type=standard" ,
"poetry run python -m fedpara.main --config-name cifar100 num_epochs=10 model.conv_type=standard")

# for loop on comm and open tmux session for each command
for i in "${comm[@]}"
do
    tmux new-session -d -s "fedpara" -n "fedpara" $i
    tmux send-keys -t "fedpara" "C-m"
done
```