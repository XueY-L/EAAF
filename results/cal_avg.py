path = '/home/yxue/EAAF/results/domainnet_bs17/DomainNet126_[\'real\', \'sketch\']_target-clipart_bs17.txt'

f = open(path, 'r')

lines = f.readlines()

sum_ = 0
for l in lines:
    sum_ += float(l)
print(sum_ / len(lines))