import pickle
import time
# import petsclinearsystem
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

zmin = -0.75
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

print("Grid dimension: [{}, {}, {}]\n".format(nW1, nW2, nW3))
print("Grid step: [{}, {}, {}]\n".format(hW1, hW2, hW3))

dVec = np.array([hW1, hW2, hW3])
increVec = np.array([1, nW1, nW1*nW2], dtype=np.int32)

petsc_mat = PETSc.Mat().create()
petsc_mat.setType('aij')
petsc_mat.setSizes([nW1 * nW2 * nW3, nW1 * nW2 * nW3])

#==============================================================================#
#    Boundary condition
#==============================================================================#

boundc = args.boundc
if boundc == 0:
    bc = "Natural"
    NNZ = 49
elif boundc == 1:
    bc = "Mix"
    NNZ = 25
elif boundc == 2:
    bc = "Neumann"
    NNZ = 25
elif boundc == 3:
    bc = "Natural Two Dimension"
    NNZ = 25

print("Boundary condition:", bc)
petsc_mat.setPreallocationNNZ(NNZ)

petsc_mat.setUp()
ksp = PETSc.KSP()
ksp.create(PETSc.COMM_WORLD)
ksp.setType('bcgs')
ksp.getPC().setType('ilu')
ksp.setFromOptions()

FC_Err = 1
epoch = 0
max_iter = args.maxiter
tol = 1e-6
fraction = args.fraction
epsilon = args.epsilon
print("epislon:", epsilon)
print("fraction:",fraction)

############ Initialization ############
V0 = W2_mat**2 - 5

i1_star = alpha/3*np.ones(W1_mat.shape)
i2_star = alpha/3*np.ones(W1_mat.shape)
h1_star = -0.025*np.zeros(W1_mat.shape)
h2_star = -0.025*np.zeros(W1_mat.shape)
hz_star = -0.025*np.zeros(W1_mat.shape)

while FC_Err > tol and epoch < max_iter:
    
    start_eps = time.time()

    dVdW1= finiteDiff_3D(V0, 0, 1, hW1)
    ddVddW1= finiteDiff_3D(V0, 0, 2, hW1)

    dVdW2 = finiteDiff_3D(V0, 1, 1, hW2)
    ddVddW2 = finiteDiff_3D(V0, 1, 2, hW2)
    
    ddVdW1dW2 = finiteDiff_3D(dVdW1, 1, 1, hW2)

##########################investment-capital ratio#############

    k1a = ((1-zeta) + zeta*np.exp(W1_mat)**(1-kappa))**(1/(kappa-1))
    k2a = ((1-zeta)*np.exp(W1_mat)**(kappa-1) + zeta)**(1/(kappa-1))

    i1_star[i1_star<=0] = 0.001
    i2_star[i2_star<=0] = 0.001

    c_star= alpha - i1_star*k1a - i2_star*k2a
    c_star[c_star<=0] = 0.001
    mc = delta * np.exp((rho-1)*V0) * c_star**(-rho)
    
    i1_new = ((1-zeta)*k1a**(1-kappa) - dVdW1) / (mc*k1a) - 1
    i1_new = i1_new/phi1

    i2_new = (zeta*k2a**(1-kappa)+ dVdW1)/(mc*k2a) -1 
    i2_new = i1_new/phi2

    i1 = i1_new * fraction + i1_star*(1-fraction)
    i2 = i2_new * fraction + i2_star*(1-fraction)
    i1[i1<=0] = 0.001
    i2[i2<=0] = 0.001

    c = alpha - i1*k1a - i2*k2a
    c[c<=0] = 0.001

########################## distortion #############

    h1_new = (1-zeta)*(k1a)**(1-kappa)*sigma_1[0] + zeta*(k2a)**(1-kappa)*sigma_2[0] + (sigma_2-sigma_1)[0]*dVdW1 + sigma_z1[0] *dVdW2
    h2_new = (1-zeta)*(k1a)**(1-kappa)*sigma_1[1] + zeta*(k2a)**(1-kappa)*sigma_2[1] + (sigma_2-sigma_1)[1]*dVdW1 + sigma_z1[1] *dVdW2
    hz_new = (1-zeta)*(k1a)**(1-kappa)*sigma_1[2] + zeta*(k2a)**(1-kappa)*sigma_2[2] + (sigma_2-sigma_1)[2]*dVdW1 + sigma_z1[2] *dVdW2

    h1 = h1_new * fraction + h1_star*(1-fraction)
    h2 = h2_new * fraction + h2_star*(1-fraction)
    hz = hz_new * fraction + hz_star*(1-fraction)

    # h1 = -h1
    # h2 = -h2
    # hz = -hz

    # h1[h1>=-1e-16] = -1e-16
    # h2[h2>=-1e-16] = -1e-16
    # hz[hz>=-1e-16] = -1e-16

########################## FDM #############

    dkadk1dk1 = (kappa-1) * ((1-zeta)**2*(k1a)**(-2*kappa+2) - kappa/(kappa-1)*(1-zeta)*(k1a)**(-kappa+1))
    dkadk1dk2 = (kappa-1) * zeta*(1-zeta)*(k1a)**(-kappa+1)*(k2a)**(-kappa+1) 
    dkadk2dk2 = (kappa-1) * (zeta**2*(k2a)**(-2*kappa+2) - kappa/(kappa-1)*(1-zeta)*(k2a)**(-kappa+1))
    
    Phi1 = 1/phi1 * np.log(1+phi1*i1)
    Phi2 = 1/phi2 * np.log(1+phi2*i2)

    A = np.zeros(W1_mat.shape)
    B_1 = Phi2 - Phi1 + (beta2-beta1)*W2_mat - eta2*np.ones(W1_mat.shape) + eta1*np.ones(W1_mat.shape) - 1/2*(np.sum(sigma_2**2)-np.sum(sigma_1**2))*np.ones(W1_mat.shape)
    B_2 = -a11*W2_mat
    B_3 = np.zeros(W1_mat.shape)
    C_1 = np.sum((sigma_2-sigma_1)**2)/2*np.ones(W1_mat.shape)
    C_2 = np.sum((sigma_z1)**2)/2*np.ones(W1_mat.shape)
    C_3 = np.zeros(W1_mat.shape)
    C_12 = np.sum(sigma_z1*(sigma_2-sigma_1))*np.ones(W1_mat.shape)
    C_23 = np.zeros(W1_mat.shape)
    C_31 = np.zeros(W1_mat.shape)
    D = delta/(1-rho) * (c**(1-rho)*np.exp((rho-1)*V0) - 1) 
    D += (1-zeta)*k1a**(1-kappa)*(Phi1+beta1*W2_mat-eta1*np.ones(W1_mat.shape)) 
    D += zeta*k2a**(1-kappa)*(Phi2+beta2*W2_mat-eta2*np.ones(W1_mat.shape))
    D += 1/2*(np.sum(sigma_1**2)*dkadk1dk1 + np.sum(sigma_2**2)*dkadk2dk2 + 2*np.sum(sigma_1*sigma_2)*dkadk1dk2)
    D += (1-gamma)/2 * np.sum(h1**2 + h2**2 + hz**2)
    
    start_ksp = time.time()

    A_1d = A.ravel(order='F')
    B_1_1d = B_1.ravel(order='F')
    B_2_1d = B_2.ravel(order='F')
    B_3_1d = B_3.ravel(order='F')
    C_1_1d = C_1.ravel(order='F')
    C_2_1d = C_2.ravel(order='F')
    C_3_1d = C_3.ravel(order='F')
    C_12_1d = C_12.ravel(order='F')
    C_23_1d = C_23.ravel(order='F')
    C_31_1d = C_31.ravel(order='F')
    D_1d = D.ravel(order='F')
    if boundc == 0:
        bc = "Natural"
        petsclinearsystemXDiff.formLinearSystem_DirectCrossDiff_Natural(W1_mat_1d, W2_mat_1d, W3_mat_1d, A_1d, B_1_1d, B_2_1d,
                                       B_3_1d, C_1_1d, C_2_1d, C_3_1d, C_12_1d, C_23_1d, C_31_1d, epsilon, lowerLims, upperLims, dVec, increVec, petsc_mat)
    elif boundc == 1:
        bc = "Mix"
        petsclinearsystemXDiff.formLinearSystem_DirectCrossDiff_Mix(W1_mat_1d, W2_mat_1d, W3_mat_1d, A_1d, B_1_1d, B_2_1d,
                                       B_3_1d, C_1_1d, C_2_1d, C_3_1d, C_12_1d, C_23_1d, C_31_1d, epsilon, lowerLims, upperLims, dVec, increVec, petsc_mat)
    elif boundc == 2:
        bc = "Neumann"
        petsclinearsystemXDiff.formLinearSystem_DirectCrossDiff_Neumann(W1_mat_1d, W2_mat_1d, W3_mat_1d, A_1d, B_1_1d, B_2_1d,
                                       B_3_1d, C_1_1d, C_2_1d, C_3_1d, C_12_1d, C_23_1d, C_31_1d, epsilon, lowerLims, upperLims, dVec, increVec, petsc_mat)
    elif boundc == 3:
        bc = "Natural Two Dimension"
        petsclinearsystemXDiff.formLinearSystem_DirectCrossDiff_Natural_Two_Dimension(W1_mat_1d, W2_mat_1d, W3_mat_1d, A_1d, B_1_1d, B_2_1d,
                                       B_3_1d, C_1_1d, C_2_1d, C_3_1d, C_12_1d, C_23_1d, C_31_1d, epsilon, lowerLims, upperLims, dVec, increVec, petsc_mat)
    
    V0_1d = V0.ravel(order='F')
    b = V0_1d  + D_1d *epsilon
    petsc_rhs = PETSc.Vec().createWithArray(b)
    x = petsc_mat.createVecRight()

    start_ksp = time.time()
    ksp.setOperators(petsc_mat)
    ksp.setTolerances(rtol=tol)
    ksp.solve(petsc_rhs, x)
    petsc_rhs.destroy()
    x.destroy()
    out_comp = np.array(ksp.getSolution()).reshape(A.shape, order="F")
    end_ksp = time.time()
    num_iter = ksp.getIterationNumber()

    dVdW1= finiteDiff_3D(out_comp, 0, 1, hW1)
    ddVddW1= finiteDiff_3D(out_comp, 0, 2, hW1)

    dVdW2 = finiteDiff_3D(out_comp, 1, 1, hW2)
    ddVddW2 = finiteDiff_3D(out_comp, 1, 2, hW2)
    
    ddVdW1dW2 = finiteDiff_3D(dVdW1, 1, 1, hW2)

    PDE_rhs = A * V0 + B_1 * dVdW1 + B_2 * dVdW2 + C_1 * ddVddW1 + C_2 * ddVddW2 + C_12 * ddVdW1dW2 + D
    PDE_Err = np.max(abs(PDE_rhs))
    FC_Err = np.max(abs((out_comp - V0) / epsilon))

    V0 = out_comp

    i1_star = i1
    i2_star = i2
    h1_star = h1
    h2_star = h2
    hz_star = hz

    epoch += 1
    
    print("V0_max,min={},{}".format(V0.max() , V0.min()))
    print("D_max,min={},{}".format(D.max() , D.min()))
    print("c_max,min={},{}".format(c.max() , c.min()))
    print("i1_max,min={},{}".format(i1.max() , i1.min()))
    print("i2_max,min={},{}".format(i2.max() , i2.min()))
    print("h1_max,min={},{}".format(h1.max() , h1.min()))
    print("h2_max,min={},{}".format(h2.max() , h2.min()))
    print("hz_max,min={},{}".format(hz.max() , hz.min()))
    print("petsc total: {:.3f}s".format(end_ksp - start_ksp))
    print("PETSc preconditioned residual norm is {:g} iterations: {}".format(
        ksp.getResidualNorm(), ksp.getIterationNumber()))
    print("Epoch {:d} (PETSc): PDE Error: {:.10f} False Transient Error: {:.10f}" .format(
        epoch, PDE_Err, FC_Err))
    print("Epoch time: {:.4f}".format(time.time() - start_eps))

res = {
    "V0": V0,
    "i1_star": i1_star,
    "i2_star": i2_star,
    "c": c,
    "k1a":k1a,
    "k2a":k2a,
    "h1_star": h1_star,
    "h2_star": h2_star,
    "hz_star": hz_star,
    "FC_Err": FC_Err,
    "PDE_rhs":PDE_rhs,
    "W1": W1,
    "W2": W2,
    "W3": W3,
}

ouputdir = "./output/"+args.output+"/"
os.makedirs(ouputdir, exist_ok=True)

with open(ouputdir + "result_frac_{}_eps_{}_rho_{}_gamma_{}_kappa_{}_zeta_{}".format(fraction,epsilon,rho,gamma,kappa,zeta), "wb") as f:
    pickle.dump(res, f)

