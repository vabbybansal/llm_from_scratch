## Test worker configs
ray job submit --address http://192.168.68.111:8265 --working-dir . -- python scripts/test_worker_config.py

- 192.168.68.111 is the worker ip

## Run worker
ray start --head --port=6379 --dashboard-host=0.0.0.0