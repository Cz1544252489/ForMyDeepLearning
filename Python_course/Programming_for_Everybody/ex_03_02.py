hrs = input("Enter Hours:")
ra =  input("Enter rate:")
try:
  h = float(hrs)
  ra = float(ra)
except:
  print("Error, please input numeric input!")
  quit()
if h <40:
  xp = h * ra
else:
  xp = 1.5 * (h-40) * ra + 40 * ra
print(xp)