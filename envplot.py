from parameters import *
import csv
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d
# from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
# ax.xaxis.set_major_locator(MultipleLocator(150))
# ax.yaxis.set_major_locator(MultipleLocator(150))




guinfo = {}   #dict key is sensor ID and two values: sensorinfo[sensorID] = [x, y]
with open("./GUInfo.csv", 'r') as file:
    csvreader = csv.reader(file)
    for row in csvreader:
        guinfo[int(row[0])] = [float(row[1]), int(row[2]),float(row[3]), float(row[4])]

# # print(len(guinfo))
# fig = plt.figure()
# ax = plt.axes(projection="3d")
# ax.set_xlim(0, int(xMaxEnv))
# ax.set_ylim(0, int(yMaxEnv))
# ax.set_zlim(0, int(zMaxEnv))
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.scatter3D([i[2] for i in guinfo.values()], [j[3] for j in guinfo.values()],0, s=2, c='black')
#
#
# # r = 23
# # p = Circle((lsuInitPos[1],lsuInitPos[0]), r, facecolor = 'b', alpha = .2, label = 'S_UAV coverage' )
# # ax.add_patch(p)
# # art3d.pathpatch_2d_to_3d(p, z=0)
# ax.scatter3D(370,430,5, c='purple', label = 'LSU') #S_UAV trajectory
# ax.scatter3D(ljuInitPos[1],ljuInitPos[0],ljuInitPos[2], s= 15,c='green', label = 'LJU') #JUAV trajectory
# ax.scatter3D(meuPos[1],meuPos[0],meuPos[2],s=15, c='red', label = 'MEU') #EUAV location
# ax.scatter3D(mjuPos[1],mjuPos[0],mjuPos[2],s=15, c='blue', label = 'MJU') #EJUAV location
# ax.legend()
# plt.show()
