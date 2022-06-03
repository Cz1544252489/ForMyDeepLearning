fname = input("Enter file name:")
fh = open(fname)
num = 0.
count = 0
for line in fh:
  if not line.startswith("X-DSPAM-Confidence:"):
    continue
  sta = line.find("0")
  num = num + float(line[sta:sta+6])
  count = count +1
print("Average spam confidence:", num/count)