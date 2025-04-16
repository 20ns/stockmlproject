def minFlipsMonoIncr(s):
    zero = 0
    one = 0
    for i in s:
        if i == "0":
            zero += 1
        else:
            one += 1
    if one == 0:
        return 0
    if zero == 0:
        return -1
    k = []
    for i in s:
        k.append(int(i))
    print(k)

    for i in k:
        if i == 0:
            pass
        if i == 1:





b = "00110"
m = "010110"
v = "00011000"

minFlipsMonoIncr(b)