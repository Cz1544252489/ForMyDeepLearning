def computepay(h, r):
  if hrs < 40:
    result = h * r
  else:
    result = (h - 40) * r * 1.5 + 40 * r
  return result

hrs = input("Enter Hours:")
rate = input("Enter Rates:")
try:
  hrs = float(hrs)
  rate = float(rate)
except:
  print("Input error")

p = computepay(hrs, rate)
print("Pay", p)