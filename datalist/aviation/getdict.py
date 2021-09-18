fil = open("syllable.txt", "r", encoding="utf-8")
text = fil.readlines()

dic = set()
for l in text:
    if len(l.split("\t")) == 2:
        l = l.split("\t")[1]
        for w in l.split(" "):
            dic.add(w)

wf = open("dic.txt", "w", encoding="utf-8")
for w in dic:
    wf.write(w)
    wf.write("\n")