import matplotlib.pyplot as plt
import re

ptn = re.compile(r'(\d+) \[D loss: ([\d\-\.]+)\] \[G loss: ([\d\-\.]+)\]')
datas = []
with open('result.txt', 'r') as f:
    for line in f:
        result = ptn.match(line)
        if result != None:
            datas.append((int(result.group(1)) - 100000, float(
                result.group(2)), float(result.group(3))))

datas = sorted(datas, key=lambda t: t[0])
fig, ax = plt.subplots()
plt.plot([t[0] for t in datas], [t[1] for t in datas], label='D', zorder=1)
plt.plot([t[0] for t in datas], [t[2] for t in datas], label='G', zorder=0)
plt.legend()
plt.savefig('loss.png')
