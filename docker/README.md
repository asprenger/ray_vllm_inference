# README Docker

## Install Docker

[How To Install and Use Docker on Ubuntu 22.04](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-22-04)

## Setup

Build container:

    docker build -t "andspr/ray-vllm-inference" -f Dockerfile .

## Run

Run container:

    docker run --gpus=all --rm -d -p 0.0.0.0:8000:8000 -p 0.0.0.0:8265:8265 --name "ray-vllm-inference" "andspr/ray-vllm-inference"

Arguments:
 * --rm Automatically remove the container when it exits
 * -d Run container in background and print container ID
 * -p port mappings

Check that container is running:

    docker ps

The Ray Serve application should now be available on the host's localhost and network interface: 

    curl --header "Content-Type: application/json" --data '{ "prompt":"The capital of France is ", "max_tokens":32 }' http://127.0.0.1:8000/generate

Log into container:

    docker exec -it "ray-vllm-inference" /bin/bash

The service logs to: `/tmp/ray/session_latest/logs/serve`.

Environment variables can be set using: `--env SERVER_PORT=1234`.

There is a warning on startup that that Ray object store is using /tmp instead of /dev/shm because the size of /dev/shm is not large enough.
You can increase /dev/shm size by passing '--shm-size=10.24gb' to 'docker run'. Make sure to set this to more than 30% of available RAM.

Start container interactively (option -it instead of -d) to show the startup logs:

    docker run -it --gpus=all --rm -p 0.0.0.0:8000:8000 -p 0.0.0.0:8265:8265 --name "ray-vllm-inference" "andspr/ray-vllm-inference"

This is useful to see any problems/warnings during startup.

Start container interactively with a shell (this does not start the service!):

    docker run -it --gpus=all --rm -d -p 0.0.0.0:8000:8000 -p 0.0.0.0:8265:8265 --name "ray-vllm-inference" "andspr/ray-vllm-inference" /bin/bash

Terminate container:

    docker kill $CONTAINER_ID

## Push image to Docker Hub

    docker login --username ${USERNAME} --password ${PASSWORD}

    docker image push ${USERNAME}/ray-vllm-inference:latest