import numpy

def get_xy_of_i(L,i): 
    # za dati indeks cvora nadji koordinate
    return i%L, i//L

def get_i_of_xy(L,x,y):
    # za date periodicne koordinate nadji indeks
    x = x%L # vrati x na opseg [0,L)
    y = y%L # vrati y na opseg [0,L)
    return y*L + x
   
def draw(L):
    # nacrtaj klaster sa vezama
    N = L**2
    for y in range(-1,L+1):
        for x in range(-1,L+1):
            print("    |\t", end=' ')
        print("")
        print("---", end=' ')
        for x in range(-1,L+1):
            print(get_i_of_xy(L,x,y),"\t---", end=' ')
        print("")
    for x in range(-1,L+1):
        print("    |\t", end=' ')
    print("")
        
def hamiltonian(L, eps, t, t2):
    # vrati hamiltonijan
    N=L*L
    H = numpy.zeros((N,N))
    for i in range(N):
        H[i,i]=eps
        x,y = get_xy_of_i(L,i)
        for d in [-1,1]: # najblizi susedi...   
            for a,b in [(x,y+d),(x+d,y)]: # po x i y osi
                j = get_i_of_xy(L,a,b) # nadji indeks suseda
                H[i,j]=H[j,i]=t # popuni clan
            for a,b in [(x+d, y+d), (x-d, y+d)]: #dijagonalni elementi
                j = get_i_of_xy(L,a,b)
                H[i,j]=H[j,i]=t2
    return H
    
L = 4

        
# nacrtamo klaster da mozemo da proverimo da li je hamiltonijan dobar
draw(L)        
    
# nacrtamo hamiltonijan
H=hamiltonian(L, 2, 1, 3)
print(H)