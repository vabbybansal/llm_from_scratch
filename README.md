## Run unit tests
python -m unittest discover -s llm_from_scratch/ -v

## Run system metrics
mactop

## HuggingFace Auth
hf auth login

## Run PreTrain
caffeinate -i python llm_from_scratch/scripts/pretrain.py