from datasets import load_dataset

# ds = load_dataset("jpacifico/French-Alpaca-dataset-Instruct-55K")
ds = load_dataset("agentic-learning-ai-lab/daily-oracle")
idx = []
for i,x in enumerate(ds['train']):
    d = x['date']
    if '2024' in d or '2025' in d:
        idx.append(i)
        s = x['title']+' '+x['text']
        s = s.replace('\n',' ')
        print(s)
print(len(idx))

