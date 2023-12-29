import math

'lsu_x, lsu_y, lsu_h, lju_x, lju_y, lju_h,lsu_tx, lju_tx,  gu_tx'
pw = [0.5, 0.3, 0.2] #priority level weight
odMin = 1 #offloading demand min
odMax = 20 #offloading demand max

xMaxEnv = 500.00
xMinEnv = 0
yMaxEnv = 500.00
yMinEnv = 0
zMaxEnv = 95.00
zMinEnv = 5

NumGUs = 50

lsuTxMax = 150
# ljuTxMax = 70
TxMin = 0

guTxMax = 50
# guTxMin = 0
lsuInitPos = [100, 250,20]
ljuInitPos = [85,85,10]
# ljuInitPos = [0,0,0]

gunewtx = 30
lsu_tx = 100
lju_tx = 50

meuPos = [200, 150, 15]
mjuPos = [150, 300, 15]
mjuTx = 50


# meuPos = [0, 0, 0]
# mjuPos = [0, 0, 0]
# meuTx = 0
# mjuTx = 0


# SpeedUAV = 15  # same for both UAVs
AntennaAngle = math.radians(60)
MinDecPow = 10 ** -7
WaveLen = 1 / 3
AntennaGain = 1
NoisePow = 10 ** -10
ChenPowGainRef = 10 ** -5

# slotMin = 1
# slotMax= 5
# angleMax = 360
# angleMin = 0
# DataLenUAVinit = 400000
# DataLenSensorinit = 80000
# DataLenSensor = 640000
# flMax = 2
# flMin = 0.1
# recT = 20
# InitEnSUAV = 50000000
# MinSUAVEnergy = 50
# InitEnJUAV = 50000000
# MinDisUAVtoUAV = 10  # to avoid collision