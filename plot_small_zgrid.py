import pickle
import time
import petsclinearsystemXDiff
from petsc4py import PETSc
import petsc4py
import os
import sys
import numpy as np
from support import *
import argparse 
sys.stdout.flush()
petsc4py.init(sys.argv)
reporterror = True
import matplotlib as mpl
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import os
import argparse


mpl.rcParams["savefig.bbox"] = "tight"
mpl.rcParams["figure.figsize"] = (32,30)
mpl.rcParams["font.size"] = 15
mpl.rcParams["legend.frameon"] = False
mpl.style.use('classic')
mpl.rcParams["lines.linewidth"] = 5

parser = argparse.ArgumentParser(description="parameters")

parser.add_argument("--boundc",type=int)

parser.add_argument("--rho", type=float)
parser.add_argument("--gamma", type=float)
parser.add_argument("--kappa",type=float)
parser.add_argument("--zeta",type=float)

parser.add_argument("--epsilon", type=float)
parser.add_argument("--fraction", type=float)
parser.add_argument("--maxiter", type=float)

parser.add_argument("--output",type=str)
parser.add_argument("--action_name",type=str)

parser.add_argument("--hW1",type=float)
parser.add_argument("--hW2",type=float)

args = parser.parse_args()


#==============================================================================#
#    PARAMETERS
#==============================================================================#

rho = args.rho
gamma = args.gamma

phi1 = 28.0
phi2 = 28.0
eta1 = 0.013
eta2 = 0.013

a11 = 0.014
alpha = 0.05
zeta = args.zeta
kappa = args.kappa

delta = 0.002 

scale = 1.32
sigma_1 = scale * np.array([.0048, 0, 0])
sigma_2 = scale * np.array([ 0, .0048, 0])
sigma_z1 = np.array([ .011*np.sqrt(5), .011*np.sqrt(5) , .025])

beta1 = 0.01
beta2 = 0.01

#==============================================================================#
#    Grids
#==============================================================================#

ymin = -np.log(20)
ymax = np.log(20)

zmin = -0.1
zmax = -zmin

kamin = -1
kamax = 1

W1_min = ymin
W1_max = ymax
hW1 = args.hW1
W1 = np.arange(W1_min, W1_max+hW1, hW1)
nW1 = len(W1)

W2_min = zmin
W2_max = zmax
hW2 = args.hW2
W2 = np.arange(W2_min, W2_max+hW2, hW2)
nW2 = len(W2)

W3_min = 0
W3_max = 1
hW3 = 0.5
W3 = np.arange(W3_min, W3_max+hW3, hW3)
nW3 = len(W3)

(W1_mat, W2_mat, W3_mat) = np.meshgrid(W1, W2, W3, indexing='ij')
stateSpace = np.hstack([W1_mat.reshape(-1, 1, order='F'), W2_mat.reshape(-1, 1, order='F'), W3_mat.reshape(-1, 1, order='F')])

W1_mat_1d = W1_mat.ravel(order='F')
W2_mat_1d = W2_mat.ravel(order='F')
W3_mat_1d = W3_mat.ravel(order='F')

lowerLims = np.array([W1.min(), W2.min(), W3.min()], dtype=np.float64)
upperLims = np.array([W1.max(), W2.max(), W3.max()], dtype=np.float64)

fraction = args.fraction
epsilon = args.epsilon

print("Grid dimension: [{}, {}, {}]\n".format(nW1, nW2, nW3))
print("Grid step: [{}, {}, {}]\n".format(hW1, hW2, hW3))

dVec = np.array([hW1, hW2, hW3])
increVec = np.array([1, nW1, nW1*nW2], dtype=np.int32)

boundc = args.boundc
if boundc == 0:
    bc = "Natural"
elif boundc == 1:
    bc = "Mix"
elif boundc == 2:
    bc = "Neumann"
elif boundc == 3:
    bc = "Natural Two Dimension"

print("Boundary condition:", bc)

datadir = "./output/"+args.output+"/"
res = pickle.load(open(datadir + "result_frac_{}_eps_{}_rho_{}_gamma_{}_kappa_{}_zeta_{}".format(fraction,epsilon,rho,gamma,kappa,zeta),"rb"))

W1 = res["W1"]
W2 = res["W2"]
i1_star = res["i1_star"]
i2_star = res["i2_star"]
k1a = res["k1a"]
k2a = res["k2a"]
h1_star = res["h1_star"]
h2_star = res["h2_star"]
hz_star = res["hz_star"]
FC_Err = res["FC_Err"]
PDE_rhs = res["PDE_rhs"]
V0 = res["V0"]
c = res["c"]

Fig_Dir = "./figure/"+args.output+"/frac_{}".format(fraction)+"/eps_{}".format(epsilon)+"/rho_{}_gamma_{}_kappa_{}_zeta_{}".format(rho,gamma,kappa,zeta)+"/"
os.makedirs(Fig_Dir, exist_ok=True)

print("max,min={},{}".format(i1_star[:,:,0].max(),i2_star[:,:,0].min()))
print("max PDE error over whole state space ",np.max(abs(PDE_rhs)))
print("max PDE error over inner state space ",np.max(abs(PDE_rhs[20:-20,20:-20,0])))

# print("d0={}".format(d_star[int(len(W1)/2),2,2]))
# print("V0={}".format(V0[int(len(W1)/2),2,2]))

plt.plot(W1,i1_star[:,int(len(W2)/2),0],label="$i1$, max = "+str(round(np.max(i1_star[:,int(len(W2)/2),0]),5))+", min = "+str(round(np.min(i1_star[:,int(len(W2)/2),0]),5)))
plt.plot(W1,i2_star[:,int(len(W2)/2),0],label="$i2$, max = "+str(round(np.max(i2_star[:,int(len(W2)/2),0]),5))+", min = "+str(round(np.min(i2_star[:,int(len(W2)/2),0]),5)))
plt.legend()
plt.xlabel('y')
plt.title('Investment-Capital Ratio, '+str(FC_Err))  
plt.xlim([-np.log(20), np.log(20)])
plt.ylim([-0.01,0.2])
plt.savefig(Fig_Dir+"iy_eps_{}_frac_{}.png".format(epsilon,fraction))
plt.close()


plt.plot(W2,i1_star[int(len(W1)/2),:,0],label="$i1$, max = "+str(round(np.max(i1_star[int(len(W1)/2),:,0]),5))+", min = "+str(round(np.min(i1_star[int(len(W1)/2),:,0]),5)))
plt.plot(W2,i2_star[int(len(W1)/2),:,0],label="$i2$, max = "+str(round(np.max(i2_star[int(len(W1)/2),:,0]),5))+", min = "+str(round(np.min(i2_star[int(len(W1)/2),:,0]),5)))
plt.legend()
plt.xlabel('z')
plt.title('Investment-Capital Ratio, '+str(FC_Err))  
plt.xlim([-0.1, 0.1])
plt.ylim([-0.01,0.2])
plt.savefig(Fig_Dir+"iz_eps_{}_frac_{}.png".format(epsilon,fraction))
plt.close()

plt.plot(W1,c[:,int(len(W2)/2),0],label="$c$, max = "+str(round(np.max(c[:,int(len(W2)/2),0]),5))+", min = "+str(round(np.min(c[:,int(len(W2)/2),0]),5)))
plt.legend()
plt.xlabel('y')
plt.title('Consumption-Capital Ratio, '+str(FC_Err))  
plt.xlim([-np.log(20), np.log(20)])
plt.ylim([-0.01,0.05])
plt.savefig(Fig_Dir+"cy_eps_{}_frac_{}.png".format(epsilon,fraction))
plt.close()

plt.plot(W2,c[int(len(W1)/2),:,0],label="$c$, max = "+str(round(np.max(c[int(len(W1)/2),:,0]),5))+", min = "+str(round(np.min(c[int(len(W1)/2),:,0]),5)))
plt.legend()
plt.xlabel('z')
plt.title('Consumption-Capital Ratio, '+str(FC_Err))  
plt.xlim([-0.1, 0.1])
plt.ylim([-0.01,0.05])
plt.savefig(Fig_Dir+"cz_eps_{}_frac_{}.png".format(epsilon,fraction))
plt.close()


plt.plot(W1,h1_star[:,int(len(W2)/2),0],label="$h1$")
plt.plot(W1,h2_star[:,int(len(W2)/2),0],label="$h2$")
plt.plot(W1,hz_star[:,int(len(W2)/2),0],label="$hz$")
plt.legend()
plt.xlabel('y')
plt.title('Distortion, '+str(FC_Err))  
# plt.xlim([-0.02, 0.02])
# plt.ylim([0.015,0.040])
plt.savefig(Fig_Dir+"hy_eps_{}_frac_{}.png".format(epsilon,fraction))
plt.close()

plt.plot(W2,h1_star[int(len(W1)/2),:,0],label="$h1$")
plt.plot(W2,h2_star[int(len(W1)/2),:,0],label="$h2$")
plt.plot(W2,hz_star[int(len(W1)/2),:,0],label="$hz$")
plt.legend()
plt.xlabel('z')
plt.title('Distortion, '+str(FC_Err))  
# plt.xlim([-0.02, 0.02])
# plt.ylim([0.015,0.040])
plt.savefig(Fig_Dir+"hz_eps_{}_frac_{}.png".format(epsilon,fraction))
plt.close()


plt.plot(W1,k1a[:,int(len(W2)/2),0],label="$k1$")
plt.plot(W1,k2a[:,int(len(W2)/2),0],label="$k2$")
plt.legend()
plt.xlabel('y')
plt.title('Capital Fraction, '+str(FC_Err))  
# plt.xlim([-0.02, 0.02])
# plt.ylim([0.015,0.040])
plt.savefig(Fig_Dir+"ky_eps_{}_frac_{}.png".format(epsilon,fraction))
plt.close()

plt.plot(W2,k1a[int(len(W1)/2),:,0],label="$k1$")
plt.plot(W2,k2a[int(len(W1)/2),:,0],label="$k2$")
plt.legend()
plt.xlabel('z')
plt.title('Capital Fraction, '+str(FC_Err))  
# plt.xlim([-0.02, 0.02])
# plt.ylim([0.015,0.040])
plt.savefig(Fig_Dir+"kz_eps_{}_frac_{}.png".format(epsilon,fraction))
plt.close()


plt.plot(W1,V0[:,int(len(W2)/2),0],label="$V$")
plt.legend()
plt.xlabel('y')
plt.title('Value Function, '+str(FC_Err))  
plt.xlim([-np.log(20), np.log(20)])
# plt.ylim([-0.01,0.0])
plt.savefig(Fig_Dir+"Vy_eps_{}_frac_{}.png".format(epsilon,fraction))
plt.close()

plt.plot(W2,V0[int(len(W1)/2),:,0],label="$V$")
plt.legend()
plt.xlabel('z')
plt.title('Value function, '+str(FC_Err))  
plt.xlim([-0.1, 0.1])
# plt.ylim([-0.01,0.0])
plt.savefig(Fig_Dir+"Vz_eps_{}_frac_{}.png".format(epsilon,fraction))
plt.close()


plt.plot(W1,PDE_rhs[:,0,0],label="$PDE error$")
plt.legend()
plt.xlabel('y')
plt.title('PDE error, interior 60, '+str(np.max(abs(PDE_rhs[20:-20,20:-20,0]))))
plt.xlim([-np.log(20), np.log(20)])
# plt.ylim([-0.01,0.0])
plt.savefig(Fig_Dir+"ey_eps_{}_frac_{}.png".format(epsilon,fraction))
plt.close()

plt.plot(W2,PDE_rhs[0,:,0],label="$PDE error$")
plt.legend()
plt.xlabel('z')
plt.title('PDE error, all, '+str(np.max(abs(PDE_rhs[:,:,0]))))  
plt.xlim([-0.1, 0.1])
# plt.ylim([-0.01,0.0])
plt.savefig(Fig_Dir+"ez_eps_{}_frac_{}.png".format(epsilon,fraction))
plt.close()


var_name = ['Investment over Capital', 'Consumption over Ka', 'Capital over Ka']

plot_row_dims      = 1
plot_col_dims      = 3

plot_color_style   = ['Viridis', 'Plasma']
plot_color_style   = ['blues','reds', 'greens']

subplot_titles = []
subplot_types = []
for row in range(plot_row_dims):
    subplot_type = []
    for col in range(plot_col_dims):
        subplot_titles.append(var_name[col])
        subplot_type.append({'type': 'surface'})
    subplot_types.append(subplot_type)
spacing = 0.1
fig = make_subplots(rows=plot_row_dims, cols=plot_col_dims, horizontal_spacing=spacing, vertical_spacing=spacing, subplot_titles=(subplot_titles), specs=subplot_types)
fig.add_trace(go.Surface(z=res['i1_star'][:,:,0].T, x=res['W1'], y=res['W2'], colorscale=plot_color_style[0], showscale=False, name= 'i1/k1', showlegend=True), row = 1, col = 1)
fig.add_trace(go.Surface(z=res['i2_star'][:,:,0].T, x=res['W1'], y=res['W2'], colorscale=plot_color_style[1], showscale=False, name= 'i2/k2', showlegend=True), row = 1, col = 1)
fig.update_scenes(dict(xaxis_title='log y', yaxis_title='z', zaxis_title='i/k', zaxis = dict(nticks=4, range=[0.0,0.1], tickformat= ".2f")), row = 1, col = 1)

fig.add_trace(go.Surface(z=res['k1a'][:,:,0].T*res['i1_star'][:,:,0].T, x=res['W1'], y=res['W2'], colorscale=plot_color_style[0], showscale=False, name= 'i1/ka', showlegend=True), row = 1, col = 2)
fig.add_trace(go.Surface(z=res['k2a'][:,:,0].T*res['i2_star'][:,:,0].T, x=res['W1'], y=res['W2'], colorscale=plot_color_style[1], showscale=False, name= 'i2/ka', showlegend=True), row = 1, col = 2)
fig.update_scenes(dict(xaxis_title='log y', yaxis_title='z', zaxis_title='i/ka', zaxis = dict(nticks=4, range=[0.0,0.1], tickformat= ".2f")), row = 1, col = 2)
fig.update_scenes(dict(aspectmode = 'cube'), row = 1, col = 2)

fig.add_trace(go.Surface(z=res['c'][:,:,0].T, x=res['W1'], y=res['W2'], colorscale=plot_color_style[2], showscale=False, name= 'c/ka', showlegend=True), row = 1, col = 3)
fig.update_scenes(dict(xaxis_title='log y', yaxis_title='z', zaxis_title='c/ka', zaxis = dict(nticks=4, range=[0.0,0.1], tickformat= ".2f")), row = 1, col = 3)

fig.update_layout(title= 'Control Variable <br><span style="font-size: 12px;"> rho = '+ str(rho)+\
              ', gamma = ' + str(gamma) + ', kappa = ' + str(kappa) + ', zeta = ' + str(kappa) +', FC error = ' + str(res['FC_Err']) + '</span>',\
              title_x = 0.5, title_y = 0.97, height=500, width=1400, title_yanchor = 'top')
fig.update_layout(margin=dict(t=75))
fig.write_json(Fig_Dir+"i_eps_{}_frac_{}.json".format(epsilon,fraction))