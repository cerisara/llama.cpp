import json

with open("MuSR/datasets/murder_mystery.json","r") as f:
    d = json.load(f)
    # print(d[0].keys())
    for i in range(len(d)):
        for j in range(len(d[i]['questions'])):
            z = d[i]['questions'][j]
            # print(z.keys())
            # print(z['question'])
            # print(z['choices'])
            # print(z['answer'])
            # print("LEN",len(d[i]['context']))

            s = "You are a detective who will solve the following mystery: "
            s += d[i]['context'] + z['question']+' '
            s += 'Generate a short reasoning in less than 100 words to prove that the murderer is '+z['choices'][1-z['answer']]+':'

            s=s.replace('\n',' ')
            print(s)
    exit()
