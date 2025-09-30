from string import printable
from time import sleep

text = 'Hello, World!'
temp = ""

for ch in text:
  for i in printable:
    """
    //THE CODE IN THIS DOC IS THE ORIGINAL SOURCE. I SIMPLIFIED IT MYSELF

    if i == ch or ch == ' ':
      sleep(0.03)
      print(temp+i)
      temp += ch
      break
    else:
      sleep(0.03)
      print(temp+i)
    """
    sleep(0.03)
    print(temp+i)
    if i == ch or ch == ' ':
      temp += ch
      break
