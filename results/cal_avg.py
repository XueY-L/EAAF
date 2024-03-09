path = '/home/yxue/EAAF/results/domainnet126_bs50/DomainNet126_[\'clipart\', \'painting\']_target-sketch.txt'

f = open(path, 'r')

lines = f.readlines()

sum_ = 0
for l in lines:
    sum_ += float(l)
print(sum_ / len(lines))