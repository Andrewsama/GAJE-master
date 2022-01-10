import random
random.seed(0)
idx_train = random.sample(range(3327), 1664)
print(len(idx_train))
idx_test = []
for i in range(3327):
    if i not in idx_train:
        idx_test.append(i)

with open('idx_train_r.txt', 'w') as fn:
    for i in idx_train:
        fn.write(str(i)+' ')

with open('idx_test_r.txt', 'w') as fn:
    for i in idx_test:
        fn.write(str(i)+' ')
print(len(idx_train)+len(idx_test))
