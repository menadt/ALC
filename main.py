# This is a sample Python script.

# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np  # Importamos librería


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def descompLU (A):
  n = A.shape[0] # Sacamos el n de la matriz
  if np.linalg.det(A)==0:
    I=np.eye(n)
    return I,A
  else:
    if n==1:
      I=np.eye(1)
      return I, A
    else:
      if A[0,0]==0:
        return np.eye(n),A
      else:
        A11=A[0,0]
        A12=A[0,1:]
        A21=A[1:,0]
        A22=A[1:, 1:]
        L=np.zeros([n,n])
        U=np.zeros([n,n])
        L21=A21/A11
        L[1:,0] = L21
        U12=A12
        resta = L21.reshape(-1, 1) @ U12.reshape(1, -1)
        LU=A22-resta
        Ele, uu=descompLU(LU)
        L[0,0]=1
        L[1:,1:]=Ele
        U[0,0]=A11
        U[0,1:]=A12
        U[1:,1:]=uu
        for i in range(n):
          if L[i,i]==0 or U[i,i]==0:
            return np.eye(n),A
        return L,U

def solve_Ly(A,b):
    res=[]
    res.append(b[0]/A[0,0])
    for i in range(1,len(A)):
        r=0
        for j in range(i):
            r=r+A[i,j]*res[j]
        sol=(b[i]-r)
        res.append(sol)

    return y
def solve_Ux(U, y):
    n = len(U)
    res = []
    res.append(y[n - 1] / U[n - 1, n - 1])
    i = n - 2
    k = 0
    while i >= 0:
        k = 0
        r = 0
        j = n - 1
        while j > i:
            r = r + U[i, j] * res[k]
            j = j - 1
            k=k+1
        sol = (y[i] - r) / U[i, i]
        res.append(sol)
        i = i - 1

    x = np.zeros(n)
    for i in range(n):
        x[i] = res.pop()
    return x

from scipy.linalg import solve_triangular as st
A=np.array([[1,-1,0,1],[0,1,4,0],[2,-1,0,-2],[-3,3,0,-1]])
L,U=descompLU(A)
b=np.array([[1,-7,-5,1]]).T
y=np.array([[1,-7,0,4]]).T
ej3=solve_Ly(L,b)
ej3b=solve_Ux(U,ej3)
print("Ya creada", st(L,b, lower=True))
print("La mia: ", ej3)

print("Ya creada", st(U,y))
print("La mia: ", ej3b)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    C = np.array([[2]])
    B = np.array([[5, 2], [3, 5]])
    D=np.array([[2, -3, 5], [6, -1, 3], [-4, 1, -2]])
    A = np.array([[2, -3, 5, 7], [6, -1, 3, 8], [-4, 1, -2, 9], [1, 2, 3, 4]])


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
