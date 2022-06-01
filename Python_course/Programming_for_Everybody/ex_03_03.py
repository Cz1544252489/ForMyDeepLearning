score = input("Enter Score: ")
try:
  score = float(score)
except:
  print("Need numeric input!")
    
if score > 1:
  print("Error, input should not greater than 1")
elif score >= 0.9:
  print("A")
elif score >= 0.8:
  print("B")
elif score >= 0.7:
  print("C")
elif score >= 0.6:
  print("D")
else:
  print("F")