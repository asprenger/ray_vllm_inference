# README.md

Launch the service on a Ray cluster on AWS. 

For more details see: [Launching Ray Clusters on AWS](https://docs.ray.io/en/latest/cluster/vms/user-guides/launching-clusters/aws.html)

## Setup

Setup the necessary AWS credentials:

    export AWS_ACCESS_KEY_ID=...
    export AWS_SECRET_ACCESS_KEY=...

Start a Ray cluster using the YAML config file:

    ray up aws_cluster_gpu.yaml

This command starts the head node of the cluster. The head node will start worker nodes using 
AWS autoscaling according to the configuration. You can view the progress of the worker node 
startup by viewing the autoscaler status on the Ray dashboard.

You can connect via the remote Ray dashboard:

    ray dashboard aws_cluster_gpu.yaml
    
This will setup the necessary port forwarding. The dashboard can be viewed by opening http://localhost:8265.

Copy the deploy config to the head node:

    ray rsync_up aws_cluster_gpu.yaml vllm_serve_config.yaml /home/ray/vllm_serve_config.yaml

Connect to the cluster head:

    ray attach aws_cluster_gpu.yaml

Start the service on the head node using the [Serve CLI](https://docs.ray.io/en/latest/serve/api/index.html#command-line-interface-cli) and a [serve configuration](https://docs.ray.io/en/latest/serve/production-guide/config.html):

    serve run vllm_serve_config.yaml

Call the service:

    curl --header "Content-Type: application/json" --data '{ "prompt":"The capital of France is ", "max_tokens":32, "temperature":0}' http://127.0.0.1:8000/generate

Tear down the cluster:

    ray down aws_cluster_gpu.yaml

Note that the instances are stopped. You need to terminate them manually.

## Advanced options

### Install command line tools on cluster

    sudo apt update
    sudo apt -y install vim curl

### Connect

Connect to Docker container:

    ssh -tt -o IdentitiesOnly=yes -i {KEYPAIR_PATH} ubuntu@{HOSTNAME} docker exec -it ray_container /bin/bash

Connect to Docker container from a terminal on the server:

    docker exec -it ray_container /bin/bash
