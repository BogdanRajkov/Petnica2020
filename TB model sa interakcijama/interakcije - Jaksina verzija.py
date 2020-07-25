import numpy

import matplotlib as mpl
from matplotlib import pyplot as plt


def get_xy_of_i(L,i): 
    # za dati indeks chvora nadji koordinate
    return i%L, i//L

def get_i_of_xy(L,x,y):
    # za date periodichne koordinate nadji indeks
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
        
def get_fock_states(Norb):
    fock_states = list(itertools.product(list(range(2)), repeat=Norb))
    fock_states = sorted(fock_states, key=lambda x: sum(x)) # sortiraj stanja po ukupnom broju chestica
    # pretvorimo u numpy.array
    for fsi,fs in enumerate(fock_states):
        fock_states[fsi] = numpy.array(list(fs)) 
    return numpy.array(fock_states) 


def get_ac_operator(i, fock_states, a_or_c='c', statistic='Fermion'):
    # napravi operator kreacije/anihilacije na chvoru i
    p = {'a':-1, 'c':1}[a_or_c]
    N,Norb = fock_states.shape
    op = numpy.zeros((N,N))
    for fs1i,fs1 in enumerate(fock_states):
        for fs2i,fs2 in enumerate(fock_states):
            diff = fs1-fs2
            nzdiff = numpy.nonzero(diff)[0]
            if nzdiff.size!=1: continue
            if diff[nzdiff[0]]!=p: continue
            if nzdiff[0]!=i: continue
            if statistic=='Fermion': sgn = (-1)**(sum(fs1[nzdiff[0]+1:]))
            else: sgn = 1
            op[fs1i,fs2i] = sgn
    return op

def get_many_body_hamiltonian_from_operators( 
    L, t, eps, V,
    statistic='Fermion'
):
    # napravi many-body hamiltonian koristecci operatore kreacije i anihilacije
    Norb = L*L
    fock_states = get_fock_states(Norb)
    N = len(fock_states)
    H = numpy.zeros((N,N))
    for i in range(Norb):
        H += eps*numpy.dot(
            get_ac_operator(i, fock_states, 'c', statistic),
            get_ac_operator(i, fock_states, 'a', statistic)
        )
        x,y = get_xy_of_i(L,i)
        for d in ([-1,1] if L>2 else [1]): # najblizhi susedi, ali pazimo na klasteru 2x2 da ne rachunamo dvaput isto
            for a,b in [(x,y+d),(x+d,y)]: # po x i y osi
                j = get_i_of_xy(L,a,b) # nadji indeks suseda
                H += t*numpy.dot(
                    get_ac_operator(i, fock_states, 'c', statistic),
                    get_ac_operator(j, fock_states, 'a', statistic)
                )
                H += 0.5*V*numpy.einsum('ij,jk,kl,lm->im',
                    get_ac_operator(i, fock_states, 'c', statistic),
                    get_ac_operator(i, fock_states, 'a', statistic),
                    get_ac_operator(j, fock_states, 'c', statistic),
                    get_ac_operator(j, fock_states, 'a', statistic)
                )
    return H

def get_block_selector( Ntot, fock_states ):
    indices = numpy.argwhere(numpy.sum(fock_states,axis=1)==Ntot)
    return slice(indices[0][0],indices[-1][0]+1,1)

######################################################

L = 2
t = -3

# # samo da proverimo da li smo sve podesili kako treba
# print "koordinate za dati indeks chvora:"
# for i in range(L*L):
#     print i,":", get_xy_of_i(L,i)   
# print "indeks chvora za date koordinate:"   
# for x in range(-1,L+1):
#     for y in range(-1,L+1):
#         print "(",x,",",y,"):",get_i_of_xy(L,x,y)
        
# nacrtamo klaster da mozhemo da proverimo da li je hamiltonijan dobar
print("klaster izgleda ovako:")
draw(L)        
print("")   
# # nacrtamo hamiltonijan
# spH = single_particle_hamiltonian(L, t, eps)
# print "jednochestichni Hamiltonijan:"
# print spH

#Fock space and many-body Hamiltonian for spinless electrons on Norb orbitals

import itertools

statistic = 'Fermion'

Norb = L*L

fock_states = get_fock_states(Norb)
#prikazhi Fokova stanja:
print("Fokova stanja:")
print(fock_states)

# print "primer operatora kreacije: na chvoru 0"
# print get_ac_operator(0, fock_states, 'c', statistic)

# print "primer operatora popunjenosti: na chvoru 0"
# print numpy.dot(
#     get_ac_operator(0, fock_states, 'c', statistic),
#     get_ac_operator(0, fock_states, 'a', statistic),
# )

# print "primer operatora hopinga: izmedju chvorova 0 i 1"
# print numpy.dot(
#     get_ac_operator(0, fock_states, 'c', statistic),
#     get_ac_operator(1, fock_states, 'a', statistic),
# )
Neps = 10
NV = 10
epss = numpy.linspace(-10,0,Neps,endpoint=True)
Vs = numpy.linspace(0,10.,NV,endpoint=True)
ground_Ntot_epsiVi = numpy.zeros((Neps,NV))
for epsi,eps in enumerate(epss):
    for Vi,V in enumerate(Vs):
        print("--------- working eps: %g V: %g"%(eps,V))
        #print "Vishechestichni Hamiltonijan pomoccu operatora kracije i anihilacije:"
        H = get_many_body_hamiltonian_from_operators(L,t,eps,V,statistic)
        #print H

        eig0s, vec0s = [], []
        for Ntot in range(Norb+1):
            slc = get_block_selector( Ntot, fock_states )
            #print slc
            Hblock = H[slc,slc]
            #print "Ntot=%d block:"%Ntot
            #print Hblock
            eigs, vecs = numpy.linalg.eigh(Hblock)
            #print "eigs:",eigs
            eig0s.append(eigs[0])
            vec0s.append(vecs[:,0])

        print("eig0s:",eig0s)
        ground_Ntot = numpy.argmin(eig0s)
        ground_Ntot_epsiVi[epsi,Vi] = ground_Ntot
        print("ground N_tot:",ground_Ntot)
        ground_state = vec0s[ground_Ntot]
        #print "ground state:",ground_state
        slc = get_block_selector( ground_Ntot, fock_states )

        n_i = numpy.zeros((Norb))
        for i in range(Norb):
            n_i_operator = numpy.dot(
                get_ac_operator(i, fock_states, 'c', statistic),
                get_ac_operator(i, fock_states, 'a', statistic),
            )
            n_i[i] = numpy.sum(numpy.diag(n_i_operator)[slc]*numpy.abs(vec0s[ground_Ntot])**2)
        print("ground state density profile:",n_i)
        print("sum(n_i): %g"%numpy.sum(n_i), end=' ')
        print(" should be:",ground_Ntot)
        
cp = plt.pcolor(Vs,epss,ground_Ntot_epsiVi)
plt.ylabel(r"$\varepsilon$",fontsize=16)
plt.xlabel(r"$V$",fontsize=16)
plt.title(r"$\langle N_{\mathrm{tot}} \rangle$",fontsize=16)
plt.colorbar(cp)
plt.show()