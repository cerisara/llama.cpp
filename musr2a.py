with open("golds.txt","r") as f: golds = f.readlines()

nok=0
for i in range(50,250):
    with open("../llamacpp/choix."+str(i),"r") as f: ls = f.readlines()
    s = ' '.join(ls)
    s = s.strip().lower()
    if golds[i][0]=='0' and s[0]=='a': nok += 1
    elif golds[i][0]=='1' and s[0]=='b': nok += 1

acc = float(nok)/200.
print("ACC",acc)

