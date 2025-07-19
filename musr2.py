import json
import random

cases = []
with open("/home/xtof/data/MuSR/datasets/murder_mystery.json","r") as f:
    d = json.load(f)
    for i in range(len(d)):
        for j in range(len(d[i]['questions'])):
            z = d[i]['questions'][j]
            s = d[i]['context'] + z['question']+' '
            s=s.replace('\n',' ')
            cases.append(s)


with open("questions.txt","w") as g:
    with open("golds.txt","w") as gl:
        for i in range(250):
            with open("/home/xtof/llamacppgerg/qwen3_14b/ok/out."+str(i),"r") as f: ok=f.readlines()
            ok = ' '.join(ok)
            with open("/home/xtof/llamacppgerg/qwen3_14b/ko/out."+str(i),"r") as f: ko=f.readlines()
            ko = ' '.join(ko)
            if random.randint(0,1) == 0: a,b,gold=ok,ko,0
            else: a,b,gold=ko,ok,1
            s = "You are a critic detective; here's the description of a mystery case: " + cases[i]
            s += "You are given two alternative reasonings A and B to solve a murder case and you must find the correct reasoning. Here are both reasonings:\n"
            s += "A: "+a+"\n"
            s += "B: "+b+"\n"
            s += "Question: which reasoning is the correct one? Answer just with the letter A or B.\n"
            s += "Answer:"

            s = s.replace('\n',' ')
            g.write(s+'\n')
            gl.write(str(gold)+'\n')

