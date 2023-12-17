 poetry run python -m fedpara.main > "cifar10noniidaug.txt" 2>&1 &
 poetry run python -m fedpara.main --config-name cifar100  > "cifar100noniidaug.txt"  2>&1 &

wait