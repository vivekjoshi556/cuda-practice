import random

count = 2000
output_file = f"inputs/inputs_{count}.txt"

with open(output_file, "w") as fp:
    fp.write(f"{count} {count}\n")
    
    lines = []
    for i in range(count):
        items = []
        for j in range(count):
            items.append(str(random.randint(0, 100)))
        lines.append(" ".join(items))
    fp.writelines("\n".join(lines))

with open(output_file, "r") as fp:
    print(fp.read(100))