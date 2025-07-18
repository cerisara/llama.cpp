from datasets import load_dataset

# ds = load_dataset("jpacifico/French-Alpaca-dataset-Instruct-55K")
ds = load_dataset("agentic-learning-ai-lab/daily-oracle")
ds = load_dataset("ZhentingNLP/mathaug-disjoint")
ds = load_dataset("ccdv/arxiv-classification")
ds = load_dataset("mteb/ArxivClassification")

idx = []
for i,x in enumerate(ds['test']):
    print("GOLD",x['label'])
    ss = x['text'].split("\n")
    s="Paper: "
    for j in range(len(ss)):
        if not '[cs.' in ss[j]: s+=ss[j].strip()+' '
    s+="Question: "
    s+="What is the previous paper talking about? Please answer just by giving the letter corresponding to one of this possible choiches: A: Artificial Intelligence; B: Computational Engineering; C: Computer Vision; D: Data Structures; E: Information Theory; F: Neural and Evolutionary; G: Programming Languages; H: Systems and Control; I: Commutative Algebra; J: Group Theory; K: Statistics Theory. Answer:"
    print(s)
