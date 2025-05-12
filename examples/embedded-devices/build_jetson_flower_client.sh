# Depending on your choice of ML framework (TF or PyTorch), the appropiate
# base image from NVIDIA will be pulled. This ensures you get the best
# performance out of your Jetson device.

BASE_PYTORCH=dustynv/l4t-pytorch:r36.2.0
BASE_TF=dustynv/l4t-tensorflow:tf2-r36.2.0
EXTRA=""

while [[ $# -gt 0 ]]; do
  case $1 in
    -p|--pytorch)
      BASE_IMAGE=$BASE_PYTORCH
      shift
      ;;
    -t|--tensorflow)
      BASE_IMAGE=$BASE_TF
      shift
      ;;
    -r|--no-cache)
      EXTRA="--no-cache"
      shift
      ;;
    -*|--*)
      echo "Unknown option $1 (pass either --pytorch or --tensorflow)"
      exit 1
      ;;
  esac
done

DOCKER_BUILDKIT=${BUILDKIT} docker build $EXTRA \
                                        --build-arg BASE_IMAGE=$BASE_IMAGE \
                                        . \
                                        -t flower_client:latest
