fname = input("Enter file name: ")
if len(fname) < 1:
    fname = "ex_08_05.txt"

fh = open(fname)
count = 0
for line in fh:
  if line.startswith('From '):
    ss = line.split()
    count = count +1
    print(ss[1])

print("There were", count, "lines in the file with From as the first word")
