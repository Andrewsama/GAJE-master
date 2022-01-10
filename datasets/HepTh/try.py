
# f = open('graph.txt', 'rb')
# edges = [i.strip().split() for i in f]
# with open('edge.txt', 'w') as fn:
#     for i in range(len(edges)):
#         fn.write(str(int(edges[i][0]))+' '+str(int(edges[i][1]))+'\n')
temp = []
with open('HepTh_features.txt', 'r') as fn:
    a = fn.readline()
    while a:
        temp.append(a)
        a = fn.readline()

with open('features.txt','w') as fn:
    for i in range(len(temp)):
        fn.write(str(i)+' '+temp[i])

