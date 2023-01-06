var1=$(nvidia-smi --query-gpu=uuid, --format=csv,noheader)
echo $var1
