from parameters import *
import math
from envplot import guinfo
import numpy as np
from scipy.stats import rayleigh

def DisSentoUAV(TrPow):
    d = math.sqrt((WaveLen ** 2 * AntennaGain * AntennaGain * TrPow) / ((4 * math.pi) ** 2 * MinDecPow))
    return d


def DisUAVtoSensor(TrPowSUAV):
    dd = WaveLen ** 2 * AntennaGain * AntennaGain * TrPowSUAV
    cc = 8 * (math.pi) ** 2 * (1 - math.cos(AntennaAngle / 2)) * MinDecPow
    ff = dd / cc
    d = math.sqrt(ff)
    return d


def SurfAreaDef(TrPowSUAV, x_UAV, y_UAV):
    minDist = min(DisSentoUAV(guTxMax), DisUAVtoSensor(TrPowSUAV))
    area = 2 * math.pi * minDist * (1 - math.cos(AntennaAngle / 2))
    radius = math.sqrt(area / math.pi)
    insidePoints = []
    for p in guinfo.values():
        x_sensor = p[2]
        y_sensor = p[3]
        if (x_sensor - x_UAV) ** 2 + (y_sensor - y_UAV) ** 2 <= radius ** 2:
            insidePoints.append([x_sensor, y_sensor])
    IDofInsideSensor = {}
    for j in insidePoints:
        for i in guinfo.keys():
            if [guinfo[i][2],guinfo[i][3]] == j:
                IDofInsideSensor[i] = j
    return IDofInsideSensor

def calActualComUAVtoGU(TrPowSUAV, gunewTX, UAVnewPos):
    coveredGU=SurfAreaDef(TrPowSUAV,UAVnewPos[0], UAVnewPos[1])
    finalGUinCOV = {}
    for i in coveredGU.keys():
        d1 = DisSentoUAV(gunewTX)
        # d1 = DisSentoUAV(gunewTX[i-1])

        d2 = DisCalcul(coveredGU[i][0], coveredGU[i][1], 0, UAVnewPos[0], UAVnewPos[1], UAVnewPos[2])
        if d1>= d2:
            finalGUinCOV[i]= coveredGU[i]
    return finalGUinCOV


def DisCalcul(x1, y1, z1, x2, y2, z2):
    d = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
    return d


def generalchanGain(x1, y1, z1, x2, y2, z2):
    return ChenPowGainRef/(DisCalcul(x1, y1, z1, x2, y2, z2)**2)

def gutoeaveschanGain(x1, y1, z1, x2, y2, z2):
    sigma = 1 / np.sqrt(np.pi / 2)
    random_samples = rayleigh.rvs(scale=sigma, size=1000)
    rayleighcoef= np.random.choice(random_samples)
    gain = ChenPowGainRef*(DisCalcul(x1, y1, z1, x2, y2, z2)**-3) * rayleighcoef
    return gain


# def Rate(guPos, GuNewTx, SerEave, jammer, jammerTx):
#
#     chanGainGu_sereav = chanGain(guPos[0], guPos[1], 0, SerEave[0], SerEave[1], SerEave[2])
#     chanGainJ_sereav = chanGain(jammer[0], jammer[1], jammer[2], SerEave[0], SerEave[1], SerEave[2])
#     rate = math.log2(1+ ((GuNewTx*chanGainGu_sereav)/(jammerTx * chanGainJ_sereav + NoisePow)))
#     return rate


def secrecyRate(guPos, GuNewTx, lsuPos):
    chanGainGu_ser = generalchanGain(guPos[0], guPos[1], 0, lsuPos[0], lsuPos[1], lsuPos[2])
    # chanGainMJ_ser = generalchanGain(mjuPos[0], mjuPos[1], mjuPos[2], lsuPos[0], lsuPos[1], lsuPos[2])
    # chanGainGu_eav = gutoeaveschanGain(guPos[0], guPos[1], 0, meuPos[0],meuPos[1], meuPos[2])
    # cahnGainLJ_eav = generalchanGain(ljuPos[0],ljuPos[1], ljuPos[2], meuPos[0],meuPos[1], meuPos[2])

    r_g_s = math.log2(1+ ((GuNewTx*chanGainGu_ser)/ NoisePow))
    r_g_e = 0
    sRate = max(r_g_s - r_g_e, 0)
    return sRate, r_g_s, r_g_e

# def paramB(flMax, recT):
#     return np.log(flMax) /recT
#
#
# def FL(CurrTime,flMax, recT, guID,prevTime, FL_val, offTime,mode=1):
#     b = paramB(flMax, recT)
#     if prevTime[guID] == 0:
#         FL = flMax
#     elif mode==1:
#         FL= FL_val[guID]
#     elif FL_val[guID] == flMax:
#         FL = flMax
#     elif (CurrTime - prevTime[guID]) <= recT:
#         FL = math.exp(b*recT)/flMax * math.exp(b*(CurrTime-prevTime[guID] - offTime[guID]))
#     return FL


# def Reward(TrPowSensor, JUAVtrPow, SUAVtraj, JUAVtraj, EUAVtraj, CurrTime, TrPowSUAV, PrevVisTimeOfsensor):
#     AllSensorReward = 0
#     CandidateSensors = SurfAreaDef(TrPowSUAV, SUAVtraj[0], SUAVtraj[1])
#     for check in CandidateSensors:
#         d1 = DisSentoUAV(TrPowSensor)
#         d2 = DisCalcul(SUAVtraj[0], SUAVtraj[1], SUAVtraj[2],
#                        CandidateSensors.get(check)[0], CandidateSensors.get(check)[1],
#                        0)
#         if d1 >= d2:
#             value = SIV(CurrTime, check, PrevVisTimeOfsensor)
#             RateSentoSUAV = RateSensortoSUAV(TrPowSensor, d2)
#
#         else:
#             value = 0
#             RateSentoSUAV = 0
#         d3 = DisSentoUAV(TrPowSensor)
#         d4 = DisCalcul(EUAVtraj[0], EUAVtraj[1], EUAVtraj[2],
#                        CandidateSensors.get(check)[0], CandidateSensors.get(check)[1],
#                        0)
#         # print(d3, d4, 'd3, d4')
#         if d3 >= d4:
#             RateSentoEUAV = math.log2(1 + (TrPowSensor * (
#                     ChenPowGainRef / d4 ** 2) / (JUAVtrPow * (
#                     ChenPowGainRef / DisCalcul(EUAVtraj[0], EUAVtraj[1], EUAVtraj[2], JUAVtraj[0], JUAVtraj[1],
#                                                JUAVtraj[2]) ** 2) + NoisePow ** 2)))
#             # print(RateSentoEUAV, 'rate2')
#         else:
#             RateSentoEUAV = 0
#         reward = value * max(0, RateSentoSUAV - RateSentoEUAV)
#         AllSensorReward += round(reward, 3)
#     return round(AllSensorReward, 3)


# def EnerFly(OldPos, NewPos):
#     movingTime = DisCalcul(OldPos[0], OldPos[1], OldPos[2], NewPos[0], NewPos[1], NewPos[2]) / SpeedUAV
#     if NewPos[2] > OldPos[2]:
#         up = 315 * (NewPos[2] - OldPos[2]) - 211.261
#         flyener = up + (308.709 * movingTime - 0.852)
#     elif NewPos[2] < OldPos[2]:
#         down = 68.956 * (OldPos[2] - NewPos[2]) - 65.183
#         flyener = down + (308.709 * movingTime - 0.852)
#     else:
#         flyener = 308.709 * movingTime - 0.852
#     return flyener, movingTime
#
#
# def EnerComm(TrPowSUAV, TrPowSensor, CandidateSensors):
#     InitTimeUAV = DataLenUAVinit / RateSensortoSUAV(TrPowSUAV, DisUAVtoSensor(TrPowSUAV))
#     UavSenCom = TrPowSUAV * InitTimeUAV
#     HovTime = 0
#     SumEn = 0
#     for check in CandidateSensors:
#         senTime = DataLenSensor / RateSensortoSUAV(TrPowSensor, DisSentoUAV(TrPowSensor))
#         HovTime += senTime
#         SumEn += TrPowSensor * senTime
#     InitTimeSensor = DataLenSensorinit / RateSensortoSUAV(guTxMax, DisSentoUAV(guTxMax))
#     SenUavCom = (guTxMax * InitTimeSensor) * len(CandidateSensors) + SumEn
#     ecomm = UavSenCom + SenUavCom
#     totalHovTime = InitTimeUAV + HovTime + InitTimeSensor
#     return ecomm, totalHovTime
#
#
# # def EnerHov(NewPosition, totalHovTime):
# #     hovener = (4.917 * NewPosition[2] + 275.204) * totalHovTime
# #     return hovener
#
#
# def SUAVEnerConsum(OldPosition, NewPosition, TrPowSUAV, TrPowSensor, CandidateSensors):
#     flyener, movingTime = EnerFly(OldPosition, NewPosition)
#     ecomm, totalHovTime = EnerComm(TrPowSUAV, TrPowSensor, CandidateSensors)
#     # hovener = EnerHov(NewPosition, totalHovTime)
#     # totalEnCon = flyener + ecomm + hovener
#     totalEnCon = flyener + ecomm
#     totalTimeCon = movingTime + totalHovTime
#     return totalEnCon, totalTimeCon
#
#
# def JUAVEnerConsum(OldPosition, NewPosition, totalTimeCon, JUAVtrPow):
#     flyener, movingTime = EnerFly(OldPosition, NewPosition)
#     timeforhov = totalTimeCon - movingTime
#     hovener = (4.917 * NewPosition[2] + 275.204) * timeforhov
#     comenergy = JUAVtrPow * timeforhov
#     totalJUAVenergy = flyener + hovener + comenergy
#     return totalJUAVenergy


