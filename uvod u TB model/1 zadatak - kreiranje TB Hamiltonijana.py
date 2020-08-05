import numpy


def get_xy_of_i(L, i):
    # za dati indeks chvora nadji koordinate
    return i % L, i // L


def get_i_of_xy(L, x, y):
    # za date periodichne koordinate nadji indeks
    x = x % L  # vrati x na opseg [0, L)
    y = y % L  # vrati y na opseg [0, L)
    return y*L + x


def draw(L):
    # nacrtaj klaster sa vezama
    N = L**2
    for y in range(-1, L+1):
        for x in range(-1, L+1):
            print("    |\t", end=' ')
        print("")
        print("---", end=' ')
        for x in range(-1, L+1):
            print(get_i_of_xy(L, x, y), "\t---", end=' ')
        print("")
    for x in range(-1, L+1):
        print("    |\t", end=' ')
    print("")


def hamiltonian(L, eps, t):
    # vrati hamiltonijan
    N = L*L
    H = numpy.zeros((N, N))
    for i in range(N):
        H[i, i] = eps
        x, y = get_xy_of_i(L, i)
        for d in [-1, 1]:  # najblizhi susedi...
            for a, b in [(x, y+d), (x+d, y)]:  # po x i y osi
                j = get_i_of_xy(L, a, b)  # nadji indeks suseda
                H[i, j] = H[j, i] = t  # popuni chlan
    return H


L = 3
# samo da proverimo da li smo sve podesili kako treba
for i in range(L*L):
    print(i, ":", get_xy_of_i(L, i))

for x in range(-1, L+1):
    for y in range(-1, L+1):
        print("(", x, ", ", y, "):", get_i_of_xy(L, x, y))

# nacrtamo klaster da mozhemo da proverimo da li je hamiltonijan dobar
draw(L)

# nacrtamo hamiltonijan
matrica = hamiltonian(L, 2, 1)
print(hamiltonian(L, 2, 1))
