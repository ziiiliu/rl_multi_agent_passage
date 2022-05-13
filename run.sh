docker run \
    --gpus all \
    --shm-size 32g \
    --volume ${PWD}/src:/home/src:Z \
    --volume ${PWD}/results:/home/results:Z \
    --volume ${PWD}/nn_results:/home/nn_results:Z \
    --volume ${PWD}/../model_ckpts:/home/model_ckpts:Z \
    -ti passage2:latest "$@"
