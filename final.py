import numpy as np
import math
import matplotlib.pyplot as plt


## We use spider web triangular mesh ##

k = 100    ## Total nodes on the surface
m = 4     ## Total nodes on each radius
r = 2       ## Radius
f = -10000  ## Force
E = 98000   ## Young Modulus
v = 0.24    ## Poisson Ratio

# First, make the total K matrix with all element 0
# The total nodes we have: m*k + 1
KE = np.zeros((2*m*k + 2, 2*m*k +2))

# The origin point
O = [[0,0]]

# Insert the points on the surface
P = []
for i in range(0, k, 1):
    P = np.insert(P, 2*i, [r * math.cos(i * 2 * math.pi / k), r * math.sin(i * 2 * math.pi / k)])

# Insert the points on the inside
for i in range(k-1, -1, -1):
    for j in range(1, m, 1):
        P = np.insert(P, 2*i, [P[2 * (i+j-1)] * (m-j) / m, P[2 * (i+j-1) + 1] * (m-j) / m])

# Make it into array of points
for i in range(0, 2*m*k, 2):
    O.append([P[i], P[i+1]])

##-----------------------------------------------------------------------##

# GROUP 1: Triangle that includes origin
A = []
i = [1, 2,2+m]
A.append(i)
for j in range(1,k-1,1):
    A.append([i[0], i[1]+ j*m, i[2]+j*m])
A.append([1, m*k +2 - m, 2])

# GROUP 2: counter clockwise 1
B = []
z = [2,3, 3+m]
B.append(z)
for j in range(1,m-1,1):
    B.append([z[0]+j, z[1]+j, z[2]+j])
for j in range(0,m-1,1):
    for l in range(1,k-1,1):
        B.append(np.ndarray.tolist(np.array(B[j]) + m*l))

z_z = [(k-1)*m + 2, (k-1)*m + 3, 3]
B.append(z_z)
for j in range(1, m-1,1):
    B.append([z_z[0]+j, z_z[1]+j, z_z[2]+j])


# GROUP 3: counter clockwise 2
C = []
y = [2, 2+m+1, 2+m]
C.append(y)
for j in range(1,m-1,1):
    C.append([y[0]+j, y[1]+j, y[2]+j])
for j in range(0,m-1,1):
    for l in range(1,k-1,1):
        C.append(np.ndarray.tolist(np.array(C[j]) + m*l))

y_y = [(k-1)*m + 2, 3, 2]
C.append(y_y)
for j in range(1, m-1,1):
    C.append([y_y[0]+j, y_y[1]+j, y_y[2]+j])

# All triangle indexes store in list I
I = A+B+C

##--------------------------------------------------------------------##

# D matrix (Plane Strain Problem)
def D_strain(Young ,Poison):
    E = Young
    v = Poison
    D = E/((1+v)*(1-2*v)) * np.array([[1-v, v, 0], [v, 1-v, 0], [0, 0, (1-2*v)/2]])
    return D

# D matrix (Plane Stress Problem)
def D_stress(Young ,Poison):
    E = Young
    v = Poison
    D = E/((1+v)*(1-v)) * np.array([[1, v, 0], [v, 1, 0], [0, 0, (1-v)/2]])
    return D

# a (Area of Triangle)
def a_matrix(x, y, z):
    x1, y1 = x
    x2, y2 = y
    x3, y3 = z
    a = 0.5 * np.linalg.det(np.array([[1, x1, y1], [1, x2, y2], [1, x3, y3]]))
    return a

# B matrix
def B_matrix(i, j, k, a):
    x1, y1 = i
    x2, y2 = j
    x3, y3 = k
    dN1_dx = (y2-y3)/(2*a)
    dN1_dy = (x3-x2)/(2*a)
    dN2_dx = (y3-y1)/(2*a)
    dN2_dy = (x1-x3)/(2*a)
    dN3_dx = (y1-y2)/(2*a)
    dN3_dy = (x2-x1)/(2*a)
    B = np.array([[dN1_dx, 0, dN2_dx, 0, dN3_dx, 0],[0, dN1_dy, 0, dN2_dy, 0, dN3_dy], \
        [dN1_dy, dN1_dx, dN2_dy, dN2_dx, dN3_dy, dN3_dx]])
    return B

def K_matrix(a, B, D):
    K = a * np.transpose(B).dot(D.dot(B))
    return K

def K(x, y, z):
    D = D_strain(E, v)
    a = a_matrix(x, y, z)
    B = B_matrix(x, y, z, a)
    K = K_matrix(a, B, D)
    return K


##-----------------------------------------------------------------------##
# Combine all the K matrix into KE
def TotalK(O, I):
    for i in range(0, k*(2*m - 1),1):
        pi = I[i][0]
        pj = I[i][1]
        pk = I[i][2]
        P1 = O[pi-1]
        P2 = O[pj-1]
        P3 = O[pk-1]
        T = [[0, 2*(pi-1)], [1, 2*(pi-1) + 1], [2, 2*(pj-1)], [3, 2*(pj-1) + 1], [4,2*(pk-1)], \
            [5,2*(pk-1) + 1]]
        for j in range(0,6):
            for l in range(0,6):
                KE[T[j][1]][T[l][1]] += K(P1,P2,P3)[T[j][0]][T[l][0]]
    return KE

# The total K Matrix
KE = TotalK(O, I)

# Reduce the matrix KE 
KE = np.delete(np.delete(KE, int(2*(0.75*k*m + m + 1))-1, 0), int(2*(0.75*k*m + m + 1))-1, 1)
KE = np.delete(np.delete(KE, int(2*(0.75*k*m + m + 1))-2, 0), int(2*(0.75*k*m + m + 1))-2, 1)
for j in range(m, 1, -1):
    KE = np.delete(np.delete(KE, int(2*(0.75*k*m + j))-2, 0), int(2*(0.75*k*m + j))-2, 1)
for j in range(m+1, 1, -1):
    KE = np.delete(np.delete(KE, int(2*(0.25*k*m + j))-2, 0), int(2*(0.25*k*m + j))-2, 1)
KE = np.delete(np.delete(KE, 0, 0), 0, 1)
##-------------------------------------------------------------------------##
# Displacement and Force matrix
F = np.zeros(2*m*k - 2*m)
F[int(2*(0.25*k*m + m + 1)) - m - 2] = f

# Getting the displacement matrix
U = np.dot(np.linalg.inv(KE), F)

# Inerting 0 displacement
U = np.insert(U, 0, 0)
for j in range(2, m+2, 1):
    U = np.insert(U, int(2*(0.25*k*m + j))-2, 0)
for j in range(2, m+1, 1):
    U = np.insert(U, int(2*(0.75*k*m + j))-2, 0)
U = np.insert(U, int(2*(0.75*k*m + m + 1))-2, 0)
U = np.insert(U, int(2*(0.75*k*m + m + 1))-1, 0)

# Reshape the matrix to get the displacement of each coordinate
U = np.reshape(U, (k*m + 1, 2))

# Matrix of node coordinates after deformation
OO = O+U

# Putting boundary of -1 to the y-coordinate
Check = []
for j in range(0, k*m + 1, 1):
    Check.append(OO[j][1])

for i in range(len(Check)):
    if Check[i] < -1 * r:
        Check[i] = -1 * r
    OO[i][1] = Check[i]
#------------------------------------------------------------------------##
# Get the tau matrix tau = D. B. u
tau = []
for i in range(0, k*(2*m - 1),1):
        pi = I[i][0]
        pj = I[i][1]
        pk = I[i][2]
        P1 = O[pi-1]
        P2 = O[pj-1]
        P3 = O[pk-1]
        a = a_matrix(P1, P2, P3)
        B = B_matrix(P1, P2, P3, a)
        Ui = U[pi-1]
        Uj = U[pj-1]
        Uk = U[pk-1]
        U_ijk = [Ui[0], Ui[1], Uj[0], Uj[1], Uk[0], Uk[1]]
        tau.append(np.ndarray.tolist(D_strain(E,v).dot(np.dot(B, np.array(U_ijk)))))

# Stress vector tau xx
tau_xx = []
for i in range(0, k*(2*m - 1),1):
    tau_xx.append(tau[i][0])
# Stress vector tau yy
tau_yy = []
for i in range(0, k*(2*m - 1),1):
    tau_yy.append(tau[i][1])
# Stress vector tau xy
tau_xy = []
for i in range(0, k*(2*m - 1),1):
    tau_xy.append(tau[i][2])

# Make each list integer
int_tau_xx = [int(item) for item in tau_xx]
int_tau_yy = [int(item) for item in tau_yy]
int_tau_xy = [int(item) for item in tau_xy]

# Find the max value of each tau
max_xx = max(int_tau_xx, key=abs)
max_yy = max(int_tau_yy, key=abs)
max_xy = max(int_tau_xy, key=abs)

# Find the index that has the max value
xx = []
for i in range(len(int_tau_xx)):
    if int_tau_xx[i] == max_xx:
            xx.append(i)
yy = []
for i in range(len(int_tau_yy)):
    if int_tau_yy[i] == max_yy:
            yy.append(i)
xy = []
for i in range(len(int_tau_xy)):
    if int_tau_xy[i] == max_xy:
            xy.append(i)

#------------------------------------------------------------------------##
#print the triangular mesh

fig = plt.figure()
ax = fig.add_subplot()
plt.grid()
ax.add_patch(plt.Circle( (0.0, 0.0 ), r ,fill = False ))
for i in range(0, k*(2*m - 1),1):
    plt.plot([OO[I[i][0]-1][0], OO[I[i][1]-1][0], OO[I[i][2]-1][0], OO[I[i][0]-1][0]], \
        [OO[I[i][0]-1][1], OO[I[i][1]-1][1], OO[I[i][2]-1][1], OO[I[i][0]-1][1]], 'b')

# Draw the triangle where the stress is highest in red
for l in range(0, len(xx), 1):
    plt.plot([OO[I[xx[l]][0]-1][0], OO[I[xx[l]][1]-1][0], OO[I[xx[l]][2]-1][0], OO[I[xx[l]][0]-1][0]], \
        [OO[I[xx[l]][0]-1][1], OO[I[xx[l]][1]-1][1], OO[I[xx[l]][2]-1][1], OO[I[xx[l]][0]-1][1]], 'r')
for l in range(0, len(yy), 1):
    plt.plot([OO[I[yy[l]][0]-1][0], OO[I[yy[l]][1]-1][0], OO[I[yy[l]][2]-1][0], OO[I[yy[l]][0]-1][0]], \
        [OO[I[yy[l]][0]-1][1], OO[I[yy[l]][1]-1][1], OO[I[yy[l]][2]-1][1], OO[I[yy[l]][0]-1][1]], 'r')
for l in range(0, len(xy), 1):
    plt.plot([OO[I[xy[l]][0]-1][0], OO[I[xy[l]][1]-1][0], OO[I[xy[l]][2]-1][0], OO[I[xy[l]][0]-1][0]], \
        [OO[I[xy[l]][0]-1][1], OO[I[xy[l]][1]-1][1], OO[I[xy[l]][2]-1][1], OO[I[xy[l]][0]-1][1]], 'r')

plt.axhline(y = -1 * r, linewidth = 2, color = 'g', linestyle = '-')
plt.show()
