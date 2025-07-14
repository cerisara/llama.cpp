from datasets import load_dataset

ds = load_dataset("jpacifico/French-Alpaca-dataset-Instruct-55K")
for x in ds['train']:
    print(x['instruction']+' : '+x['output'])

