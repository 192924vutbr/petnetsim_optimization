
import random, string
p=10

t=5
div = 3
places = []
trans=[]


for i in range(p):
    places.append(str(''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(8))))
for i in range(t):
    trans.append(str("TransitionStochastic('" + ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(8)) + "')"))





''

print(places)
print(trans)