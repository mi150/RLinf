#!/bin/bash

# Parameter check
if [ -z "$RANK" ]; then
    echo "Error: RANK environment variable not set!"
    exit 1
fi

# Configuration file path (modify according to actual needs)
SCRIPT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_PATH=$(dirname "$SCRIPT_PATH")
RAY_HEAD_IP_FILE=$REPO_PATH/ray_utils/ray_head_ip.txt
RAY_PORT=${RAY_PORT:-8888}  # Default port for Ray, can be modified if needed

# Head node startup logic
if [ "$RANK" -eq 0 ]; then
    # Get local machine IP address (assumed to be intranet IP)
    IP_ADDRESS=$(hostname -I | awk '{print $1}')
    IP_ADDRESS="166.111.249.22"
    # Start Ray head node
    echo "Starting Ray head node on rank 0, IP: $IP_ADDRESS"
    ray start --head --memory=400000000000 --port=$RAY_PORT --node-ip-address=$IP_ADDRESS --dashboard-host=0.0.0.0 --dashboard-port=8891      --object-manager-port=8893 --runtime-env-agent-port=8894 --dashboard-agent-grpc-port=8895 --metrics-export-port=8896 --node-manager-port=8892 --min-worker-port=8900 --max-worker-port=8999 --dashboard-agent-listen-port=8897
    
    # Write IP to file
    echo "$IP_ADDRESS" > $RAY_HEAD_IP_FILE
    echo "Head node IP written to $RAY_HEAD_IP_FILE"
else
    # Worker node startup logic
    echo "Waiting for head node IP file..."
    
    # Wait for file to appear (wait up to 360 seconds)
    for i in {1..360}; do
        if [ -f $RAY_HEAD_IP_FILE ]; then
            HEAD_ADDRESS=$(cat $RAY_HEAD_IP_FILE)
            if [ -n "$HEAD_ADDRESS" ]; then
                break
            fi
        fi
        sleep 1
    done
    if [ -z "$HEAD_ADDRESS" ]; then
        echo "Error: Could not get head node address from $RAY_HEAD_IP_FILE"
        exit 1
    fi
    
    echo "Starting Ray worker node connecting to head at $HEAD_ADDRESS"
    ray start --memory=400000000000 --address="$HEAD_ADDRESS:$RAY_PORT" --node-ip-address="101.126.84.211"  --node-manager-port=8892 --object-manager-port=8893 \
    --runtime-env-agent-port=8894 --dashboard-agent-grpc-port=8895 --metrics-export-port=8896 --dashboard-agent-listen-port=8897 --min-worker-port=8900  --max-worker-port=8999
fi