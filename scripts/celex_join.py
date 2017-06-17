import sys

out = open("joined.txt", "w")
lems = open(sys.argv[1], "r")
segs = open(sys.argv[2], "r")
for l,s in zip(lems.readlines(), segs.readlines()):
    l = l.strip()
    s = s.strip()
    if l == s: continue
    if len(l) == 0 or len(s) == 0: continue
    if len(l.split()) > 1: continue # don't support multi-words
    out.write(l + " " + " ".join(s.split("+")) + "\n")
out.close()
