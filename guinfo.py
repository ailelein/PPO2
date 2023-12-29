import random
import csv
from parameters import *
with open('GUInfo.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    for s in range(1, NumGUs+1):
        b= float(random.choice([pw[0],pw[1],pw[2]])) #priority weights
        c= int(random.randint(odMin, odMax)) # offloading data
        d=round(random.uniform(1.00, xMaxEnv-1), 2)
        e = round(random.uniform(1.00, yMaxEnv-1), 2)
        a = [s, b, c, d, e]
        writer.writerow(a)
file.close()
