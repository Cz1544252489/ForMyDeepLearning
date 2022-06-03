text = "X-DSPAM-Confidence:    0.8475"
tmp = text.find('0')
print(float(text[tmp:tmp+6]))