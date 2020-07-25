import numpy as np
import itertools

t=-1
eps=2
L=2

def get_x_of_i(L,i): 
    # za dati indeks chvora nadji koordinate
    return i

def get_i_of_x(L, x):
    # za date periodichne koordinate nadji indeks
    x = x%L # vrati x na opseg [0,L)
    return x
    
def draw(L):
    # nacrtaj klaster sa vezama
    print("---", end="")
    for x in range(-1,L+1):
        print(get_i_of_x(L,x),"\t---", end=" ")
    print()
        
def single_particle_hamiltonian(L, t, eps):
    # vrati hamiltonijan
    H = np.zeros((L, L))
    for i in range(L):
        H[i,i]=eps
        for d in [-1,1]: # najblizhi susedi...   
            for a in [i+d]: # po x i y osi
                j = get_i_of_x(L,a) # nadji indeks suseda
                H[i,j]=H[j,i]=t # popuni chlan
    return H

singleHamiltonian=single_particle_hamiltonian(L, -1, 2)
print(singleHamiltonian)


def get_fock_states(max_occupancy, Norb):
    fock_states = list(itertools.product(range(max_occupancy+1), repeat=Norb))
    fock_states = sorted(fock_states, key=lambda x: sum(x)) # sortiraj stanja po ukupnom broju chestica
    # pretvorimo u numpy.array
    for fsi,fs in enumerate(fock_states):
        fock_states[fsi] = np.array(list(fs)) 
    return np.array(fock_states) 

    return fock_states

fokova_stanja=get_fock_states(2, L)
print(fokova_stanja)

def many_body_hamiltonian(fock_states, spH):
     N = len(fock_states)
     H = np.zeros((N,N))
     for fsi,fs in enumerate(fock_states):
         for i,n in enumerate(fs):
             H[fsi,fsi]+=spH[i,i]*n #dijagonalni element je suma potencijala svih popunjenih orbitala
     # sada popunimo vandijagonalne elemente
     for fs1i,fs1 in enumerate(fock_states):
         ntot1 = np.sum(fs1)
         for fs2i,fs2 in enumerate(fock_states):
             if fs1i<=fs2i: continue # gledamo samo gornji trougao, donji popunjavamo po simetriji
             ntot2 = np.sum(fs2)
             if ntot1!=ntot2: continue # ako dva stanja nemaju isti broj chestica, nisu u vezi
             diff = fs1-fs2
             nzdiff = np.nonzero(diff)[0]
             if nzdiff.size>2: continue # ako se pomerilo vishe od jedne chestice, to nas ne interesuje
             if np.sum(np.abs(diff)) > 2: continue
             pref = np.sqrt(fs2[nzdiff[1]])*np.sqrt(fs1[nzdiff[0]])
             H[fs1i,fs2i]=H[fs2i,fs1i]=pref*spH[tuple(nzdiff)] # nadjeno hoping amplitudu izmedju dva stanja izmedju kojih se mrdnula chestica

     return H


print('Visecesticni Hamiltonijan')
visecesticniH=many_body_hamiltonian(fokova_stanja, singleHamiltonian)
print(np.around(visecesticniH, 2))
'''
def get_ac_operator(i, fock_states, a_or_c='c', ):
    # napravi operator kreacije/anihilacije na chvoru i
    if a_or_c=='a': #izaberi c za kreacioni ili a za anihilacioni operator
        p=-1
    elif a_or_c=='c':
        p=1 
    N,Norb = fock_states.shape #N je broj mogucih stanja, Norb je broj orbitala tj. broj cvorova
    op = np.zeros((N,N)) #kreacioni/anihilacioni operator u zavisnosti od izbora a ili c
    for fs1i,fs1 in enumerate(fock_states):
        for fs2i,fs2 in enumerate(fock_states): #ove dve for petlje sluze da uporede svako sa svakim fokovim stanjem
            diff = fs1-fs2 #oduzima fokova stanja i vraca niz. Ukoliko niz sadrzi 1 to se pise u kreacioni, a ako sadrzi -1 pise se u anihi
            nzdiff = np.nonzero(diff)[0] #mesto clana u nizu diff koji nije 0 (nonzero). npr nzdiff=3 i onda je diff[3]=1
            if nzdiff.size!=1: continue #ne zanima nas slucaj kada je kreirano/unisteno vise od jedne cestice
            if diff[nzdiff[0]]!=p: continue #proverava da li je dodata ili oduzeta cestica izmedju dva fokova stanja
            if nzdiff[0]!=i: continue #nas interesuje samo stanje na cvoru i
            op[fs1i,fs2i] = 1 #popunjavanje operatora
    return op
'''

def get_ac_operator(i, fock_states, max_occupancy, a_or_c='c'):
     # napravi operator kreacije/anihilacije na chvoru i
     if a_or_c=='c':
         p=1
     elif a_or_c=='a':
         p=-1
     else: assert False, "unknown type of operator"
     N,Norb = fock_states.shape
     op = np.zeros((N,N))
     for fs1i,fs1 in enumerate(fock_states):
         if fs1[i]+p not in range(max_occupancy+1): continue #preskachemo ako ispadne iz opsega
         fs2 = np.array(fs1)
         fs2[i] += p
         fs2i = np.nonzero((fock_states == fs2).all(axis=1))[0][0]
         if a_or_c=='c':
             term = np.sqrt(fs2[i])
         elif a_or_c=='a':
             term = np.sqrt(fs1[i])
         else: assert False, "unknown type of operator"
         op[fs2i, fs1i] = term
     return op

operatorC=get_ac_operator(0, fokova_stanja, 2, 'c')
#print(operatorC)
operatorA=get_ac_operator(0, fokova_stanja, 2, 'a')
#print(operatorA)

def get_tight_binding_many_body_hamiltonian_from_operators(
     L, max_occupancy, t, eps
):
     # napravi many-body hamiltonian koristecci operatore kreacije i anihilacije
     fock_states = get_fock_states(max_occupancy, L)
     N = len(fock_states)
     H = np.zeros((N,N))
     for i in range(L):
         H += eps*np.dot(
             get_ac_operator(i, fock_states, max_occupancy, 'c'),
             get_ac_operator(i, fock_states, max_occupancy, 'a')
         )
         for d in ([-1,1] if L>2 else [1]):
             j = get_i_of_x(L,i+d) # nadji indeks suseda
             H += t*np.dot(
                 get_ac_operator(i, fock_states, max_occupancy, 'c'),
                 get_ac_operator(j, fock_states, max_occupancy, 'a')
             )
     return H
 
print('Hamiltonijan dobijen mnozenjem operatora')
hamiltnojanOdOperatora=get_tight_binding_many_body_hamiltonian_from_operators(L, 2, t, eps)
print(np.around(hamiltnojanOdOperatora, 2))