from scipy import zeros, dot, random, mat, linalg, diag, sqrt, sum, hstack, ones
from scipy.linalg import norm, eig

def multMv(A,v):
    return A*v

def arnoldi(A, v0, k):
    """
    Arnoldi algorithm (Krylov approximation of a matrix)
        input: 
            A: matrix to approximate
            v0: initial vector (should be in matrix form) 
            k: number of Krylov steps 
        output: 
            V: matrix (large, N*k) containing the orthogonal vectors
            H: matrix (small, k*k) containing the Krylov approximation of A

    Author: Vasile Gradinaru, 14.12.2007 (Rennes)
    """
    #print 'ARNOLDI METHOD'
    inputtype = A.dtype.type
    V = mat( v0.copy() / norm(v0), dtype=inputtype)
    H = mat( zeros((k+1,k), dtype=inputtype) )
    for m in xrange(k):
        vt = A*V[ :, m]
        for j in xrange( m+1):
            H[ j, m] = (V[ :, j].H * vt )[0,0]
            vt -= H[ j, m] * V[:, j]
        H[ m+1, m] = norm(vt);
        if m is not k-1:
            V =  hstack( (V, vt.copy() / H[ m+1, m] ) ) 
    return V,  H

def arnoldi_fast(A, v0, k):
    """
    Arnoldi algorithm (Krylov approximation of a matrix)
        input: 
            A: matrix to approximate
            v0: initial vector (should be in matrix form) 
            k: number of Krylov steps 
        output: 
            V: matrix (large, N*k) containing the orthogonal vectors
            H: matrix (small, k*k) containing the Krylov approximation of A

    Author: Vasile Gradinaru, 14.12.2007 (Rennes)
    """
#    print 'ARNOLDI METHOD'
    inputtype = A.dtype.type
    V = mat(zeros((v0.shape[0],k+1), dtype=inputtype))
    V[:,0] = v0.copy()/norm(v0)
    H = mat( zeros((k+1,k), dtype=inputtype) )
    for m in xrange(k):
        vt = A*V[ :, m]
        for j in xrange( m+1):
            H[ j, m] = (V[ :, j].H * vt )[0,0]
            vt -= H[ j, m] * V[:, j]
        H[ m+1, m] = norm(vt);
        V[:,m+1] = vt.copy() / H[ m+1, m]
    return V,  H

def arnoldi_fast_nocopy(A, v0, k):
    """
    Arnoldi algorithm (Krylov approximation of a matrix)
    Uses in-place computations and row-major format
        input: 
            A: matrix to approximate
            v0: initial vector (should be in matrix form) 
            k: number of Krylov steps 
        output: 
            V: matrix (large, N*k) containing the orthogonal vectors
            H: matrix (small, k*k) containing the Krylov approximation of A

    Author: Vasile Gradinaru, 14.12.2007 (Rennes)
    """
#    print 'ARNOLDI METHOD'
    inputtype = A.dtype.type
    n = v0.shape[0]
    V = zeros((k+1,n), dtype=inputtype)
    V[0,:] = v0.T.copy()/norm(v0)
    H = zeros((k+1,k), dtype=inputtype)
    for m in xrange(k):
        V[m+1,:] = dot(A,V[m,:])
        for j in xrange( m+1):
            H[ j, m] = dot(V[j,:], V[m+1,:])
            V[m+1,:] -= H[ j, m] * V[j,:]
        H[ m+1, m] = norm(V[m+1,:]);
        V[m+1,:] /= H[ m+1, m]
    return V.T,  H

def lanczos(A, v0, k):
    """
    Lanczos algorithm (Krylov approximation of a matrix)
        input: 
            A: matrix to approximate
            v0: initial vector (should be in matrix form) 
            k: number of Krylov steps 
        output: 
            V: matrix (large, N*k) containing the orthogonal vectors
            H: matrix (small, k*k) containing the Krylov approximation of A
            
    Author: Vasile Gradinaru, 14.12.2007 (Rennes)
    """
    print 'LANCZOS METHOD !'
    V = mat( v0.copy() / norm(v0) )
    alpha =  zeros(k)  
    beta =  zeros(k+1)  
    for m in xrange(k):
        vt = multMv( A ,  V[ :, m])
        #vt = A * V[ :, m]
        if m > 0: vt -= beta[m] * V[:, m-1]
        alpha[m] = (V[:, m].H * vt )[0, 0]
        vt -= alpha[m] * V[:, m]
        beta[m+1] = norm(vt)
        V =  hstack( (V, vt.copy() / beta[m+1] ) ) 
    rbeta = beta[1:-1]    
    H = diag(alpha) + diag(rbeta, 1) + diag(rbeta, -1)
    return V,  H

def orlancz(A, v0, k):
    """
    full orthogonalized Lanczos algorithm (Krylov approximation of a matrix)
        input: 
            A: matrix to approximate
            v0: initial vector (should be in matrix form) 
            k: number of Krylov steps 
        output: 
            V: matrix (large, N*k) containing the orthogonal vectors
            H: matrix (small, k*k) containing the Krylov approximation of A
            
    Author: Vasile Gradinaru, 21.10.2008 (Zuerich)
    """
    print 'FULL ORTHOGONAL LANCZOS METHOD !'
    from numpy import finfo, sqrt
    reps = 10*sqrt(finfo(float).eps)
    V = mat( v0.copy() / norm(v0) )
    alpha =  zeros(k)  
    beta =  zeros(k+1)  
    for m in xrange(k):
        #vt = A * V[ :, m]
        vt = multMv( A , V[ :, m])
        if m > 0: vt -= beta[m] * V[:, m-1]
        alpha[m] = (V[:, m].H * vt )[0, 0]
        vt -= alpha[m] * V[:, m]
        beta[m+1] = norm(vt)
        # reortogonalization
        h1 = multMv(V.H, vt)
        vt -= multMv(V, h1)
        if norm(h1) > reps: vt -= multMv(V, (multMv(V.H,vt)))
        #
        V =  hstack( (V, vt.copy() / beta[m+1] ) ) 
    rbeta = beta[1:-1]    
    H = diag(alpha) + diag(rbeta, 1) + diag(rbeta, -1)
    return V,  H
    
 # ---------------------------------------------------------------------------------
    
    
    
def delta(i,j):
    if i==j: return 1
    else: return 0

def ConstructMatrix(N):
    H = mat(zeros([N,N]))
    for i in xrange(N):
        for j in xrange(N):
            #H[i,j] = 1L * sqrt((i+1)**2+(j+1)**2)+(i+1)*delta(i,j)
            H[i, j] = float( 1+min(i, j) )
    print H
    return H

def eigenvalues(H):
    return eig(H)[0]

def diagonalize(H):
    return diag(eigenvalues(H))

def randomvector(N):
    v = random.random(N)
    n = sqrt( sum(v*v) )
    return mat(v/n).T * 1L

def print_first_N(a,N):
    try: acopy = a.copy()
    except: acopy = a[:]
    acopy.sort()
    max = min(N,len(a))
    for i in range(max):
        print '%1.5g\t' % acopy[i],
    print ''

def print_first_last(a):
    print '%1.9g\t' % a.min(),
    print '%1.9g' % a.max()
    

def checkconvergence(N=10,N_to_display=5,  method=arnoldi):
    #checks convergence of lanczos approximation to eigenvalues
    H = ConstructMatrix(N)
    H = H + 1j*H
    H = H.H * H
    True_eigvals = eigenvalues(H)

    print 'True ',
    print_first_last(True_eigvals)
    

  #  v = [mat(zeros(N)).T, randomvector(N)]
    v = randomvector(N)
    #V, h = arnoldi(H, v, N)
    #V, h  = lanczos(H, v, N)
    V,  h = method(H, v, N)
 #   print 'V=',  V
    for i in xrange(1,N+1):
        print '%i    ' % i,
        #print_first_last(eigenvalues(h[:i,:i]))
        print_first_N( eigenvalues(h[:i,:i]) ,  i)
    print 'eigenvalues via eig(flapack)'
    print_first_N( True_eigvals ,  N) 


if __name__ == "__main__":
#    N = 10
#    A = ConstructMatrix(N)
##    print_first_N(eigenvalues(A),  N)
#    v = mat( ones(N) ).T
#    V, h = arnoldi(A, v, N)
##h = lanczos(H,N)
#    for i in xrange(1,N+1):
#        print '%i    ' % i,
#        #print_first_last(eigenvalues(h[:i,:i]))
#        print_first_N( eigenvalues(h[:i,:i]) ,  i)
#    print 'eigenvalues via eig(flapack)'
#    print_first_N( eigenvalues(A) ,  N) 
    checkconvergence(20,2, method=arnoldi)
    checkconvergence(20,2, method=lanczos)
    checkconvergence(20,2, method=orlancz)

    
