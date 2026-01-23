out = []
for i in range(1, 6):
    if i % 2 == 1:
        for j in range(i, 0, -2):
            out.append(j)

print(out)