# For docker
docker run -it --name h100_zz --gpus all \
    --shm-size 32g \
    --network host \
    -v /.cache:/root/.cache \
    -v <path_to_your_workspace>:/workspace \
    --env "HF_TOKEN=" \
    --ipc=host \
    lmsysorg/sglang:latest \
    /bin/bash

