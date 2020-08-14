import numpy as np
import numpy.linalg as la
import itertools
import matplotlib.pyplot as plt


def set_parameters(parameters):
    global default_parameters
    default_parameters = parameters


def get_parameter(name=None):
    global default_parameters
    if name is None:
        return default_parameters
    else:
        return default_parameters[name]


def bra(vec):
    return np.conj(vec.reshape(1, -1))


def ket(vec):
    return vec.reshape(-1, 1)


def get_xy_of_i(L, i):
    # za dati indeks chvora nadji koordinate
    return i % L, i//L


def get_i_of_xy(L, x, y):
    # za date periodichne koordinate nadji indeks
    x = x % L  # vrati x na opseg [0,L)
    y = y % L  # vrati y na opseg [0,L)
    return y*L + x


def calc_exp(operator, vec):
    exp_value = bra(vec) @ (operator @ ket(vec))
    return np.float32(exp_value[0][0])


# NE RADI!!!!! Prekopiraj onaj Jaksin za 2d sistem
def draw_cluster(parameters=None):
    if parameters is None:
        parameters = get_parameter()
    L = parameters['L']

    print('---', end='')
    for x in range(-1, L+1):
        print(x % L, '\t---', end=' ')
    print()


def construct_single_particle_hamiltonian(parameters=None):
    if parameters is None:
        parameters = get_parameter()
    L = parameters['L']
    n_orb = parameters['n_orb']
    out = np.zeros((n_orb, n_orb))

    for i in range(n_orb):
        out[i, i] = parameters['eps'] + parameters['noise'][i]
        x, y = get_xy_of_i(L, i)
        for d in [-1, 1]:  # najblizhi susedi...
            for a, b in [(x, y+d), (x+d, y)]:  # po x i y osi
                j = get_i_of_xy(L, a, b)  # nadji indeks suseda
                out[i, j] = out[j, i] = parameters['t']  # popuni chlan
    return out


def get_fock_states(parameters=None):
    if parameters is None:
        parameters = get_parameter()

    fock_states = list(
        itertools.product(range(parameters['max_occupancy'] + 1),
                          repeat=parameters['L']**2))
    # sortiraj stanja po ukupnom broju chestica
    fock_states = sorted(fock_states, key=lambda x: sum(x))
    # pretvorimo u numpy.array
    for fsi, fs in enumerate(fock_states):
        fock_states[fsi] = np.array(list(fs))
    return np.array(fock_states)

    return fock_states


'''
def construct_many_body_hamiltonian(parameters=None):
    if parameters is None:
        parameters = get_parameter()
    L = parameters['L']

    fock_states = get_fock_states(parameters)
    single_ptcl_hmltn = construct_single_particle_hamiltonian(parameters)
    n_dim = (parameters['max_occupancy'] + 1) ** L
    out = np.zeros((n_dim, n_dim))
    for fsi, fs in enumerate(fock_states):
        for i, n in enumerate(fs):
            out[fsi, fsi] += single_ptcl_hmltn[i, i] * n
            # dijagonalni element je suma potencijala svih popunjenih orbitala

    # sada popunimo vandijagonalne elemente
    for fs1i, fs1 in enumerate(fock_states):
        ntot1 = np.sum(fs1)
        for fs2i, fs2 in enumerate(fock_states):
            # gledamo samo gornji trougao, donji popunjavamo po simetriji
            if fs1i <= fs2i:
                continue
            ntot2 = np.sum(fs2)
            # ako dva stanja nemaju isti broj chestica, nisu u vezi
            if ntot1 != ntot2:
                continue
            diff = fs1-fs2
            nzdiff = np.nonzero(diff)[0]
            # ako se pomerilo vishe od jedne chestice, to nas ne interesuje
            if nzdiff.size > 2:
                continue
            if np.sum(np.abs(diff)) > 2:
                continue
            pref = np.sqrt(fs2[nzdiff[1]]) * np.sqrt(fs1[nzdiff[0]])
            out[fs1i, fs2i] = out[fs2i, fs1i] = \
                pref * single_ptcl_hmltn[tuple(nzdiff)]
            # nadjeno hoping amplitudu izmedju dva stanja
            # izmedju kojih se mrdnula chestica

    for orb1 in range(L):
        orb2 = (orb1 + 1) % L
        # biramo susede chestica, ali tako da svaki par izaberemo samo jednom
        for fsi, fs in enumerate(fock_states):
            out[fsi][fsi] += parameters['V'] * fs[orb1] * fs[orb2]

    return out
'''


def get_ac_operator(i, a_or_c, parameters=None):
    # napravi operator kreacije/anihilacije na chvoru i
    if parameters is None:
        parameters = get_parameter()
    max_occupancy = parameters['max_occupancy']
    statistic = parameters['statistic']

    if a_or_c not in ('a', 'c'):
        raise ValueError('unknown type of operator')
    if statistic not in ('Fermion', 'Boson'):
        raise ValueError('unknown statistic')

    if a_or_c == 'c':
        shift = 1
    elif a_or_c == 'a':
        shift = -1

    fock_states = get_fock_states(parameters)
    n_dim = len(fock_states)
    out = np.zeros((n_dim, n_dim))
    for fs1i, fs1 in enumerate(fock_states):
        fs2 = np.array(fs1)
        fs2[i] += shift
        if fs2[i] not in range(max_occupancy+1):
            continue  # preskachemo ako ispadne iz opsega
        fs2i = np.nonzero((fock_states == fs2).all(axis=1))[0][0]

        # if a_or_c=='c':
        #     term = np.sqrt(fs2[i])
        # elif a_or_c=='a':
        #     term = np.sqrt(fs1[i])
        # out[fs2i, fs1i] = term

        if statistic == 'Fermion':
            sgn = (-1)**(sum(fs2[i+1:]))
        else:
            sgn = 1
        out[fs2i, fs1i] = sgn
    return out


def get_state_ac_operator(state, a_or_c, parameters=None):
    if parameters is None:
        parameters = get_parameter()
    n_orb = parameters['L'] ** 2
    n_dim = (parameters['max_occupancy'] + 1) ** (parameters['L'] ** 2)

    orb_ops = np.empty((n_orb, n_dim, n_dim))
    for it in range(n_orb):
        orb_ops[it, :, :] = get_ac_operator(it, a_or_c, parameters)

    state_op = np.zeros((n_dim, n_dim))
    for it in range(n_orb):
        state_op += state[it] * orb_ops[it, :, :]

    return state_op


def get_number_operator(i, parameters=None):
    if parameters is None:
        parameters = get_parameter()

    a_op = get_ac_operator(i, 'a', parameters)
    c_op = get_ac_operator(i, 'c', parameters)

    return c_op @ a_op


def construct_hamiltonian_from_operators(parameters=None):
    # napravi many-body hamiltonian koristecci operatore kreacije i anihilacije
    if parameters is None:
        parameters = get_parameter()
    L = parameters['L']
    Norb = L*L
    fock_states = get_fock_states(parameters)
    n_dim = len(fock_states)
    out = np.zeros((n_dim, n_dim))
    a = np.zeros((Norb, n_dim, n_dim))
    c = np.zeros((Norb, n_dim, n_dim))
    for i in range(Norb):
        a[i, :, :] = get_ac_operator(i, 'a', parameters)
        c[i, :, :] = get_ac_operator(i, 'c', parameters)
        # print('Procenat izvrsenja prvog fora je ' + str((i+1)*100/Norb))

    for i in range(Norb):
        out += (parameters['eps'] + parameters['noise'][i]) * c[i] @ a[i]
        x, y = get_xy_of_i(L, i)
        # najblizhi susedi, ali pazimo na
        # klasteru 2x2 da ne rachunamo dvaput isto
        for d in ([-1, 1] if L > 2 else [1]):
            for it1, b in [(x, y+d), (x+d, y)]:  # po x i y osi
                j = get_i_of_xy(L, it1, b)  # nadji indeks suseda
                out += parameters['t'] * c[i] @ a[j]
                # out += parameters['V'] * (c[i] @ a[i]) @ \
            # (c[(i+1) % Norb] @ a[(i+1) % Norb])

                out += 0.5 * parameters['V']*np.einsum('ij,jk,kl,lm->im',
                                                       get_ac_operator(i, 'c'),
                                                       get_ac_operator(i, 'a'),
                                                       get_ac_operator(j, 'c'),
                                                       get_ac_operator(j, 'a')
                                                       )
        # print('Procenat izvrsenja DRUGOG fora je ' + str((i+1)*100/Norb))
    return out


def split_hamiltonian_into_blocks(parameters=None):
    if parameters is None:
        parameters = get_parameter()
    n_orb = parameters['n_orb']

    fock_states = get_fock_states(parameters)
    many_body_hmltn = construct_hamiltonian_from_operators(parameters)
    out = [0] * n_orb
    for i in range(n_orb):
        ind = np.nonzero(np.sum(fock_states, axis=1) == i+1)[0]
        ind2d = tuple(np.meshgrid(ind, ind))
        out[i] = many_body_hmltn[ind2d]
    return out


def find_lowest_lying_eigenstate(parameters=None):
    if parameters is None:
        parameters = get_parameter()
    n_orb = parameters['n_orb']
    fock_states = get_fock_states(parameters)
    n_dim = len(fock_states)
    split_hmltn = split_hamiltonian_into_blocks(parameters)

    eigvals, eigvecs = [0] * n_orb, [0] * n_orb
    for it in range(n_orb):
        eigvals[it], eigvecs[it] = la.eigh(split_hmltn[it])
    block_ll_eigvals = [el[0] for el in eigvals]
    ll_eigval = min(block_ll_eigvals)
    ll_eigval_ind = block_ll_eigvals.index(ll_eigval)
    ll_eigvec_trim = eigvecs[ll_eigval_ind][:, 0]
    # ovo je ll eigenvector, ali je trenutno samo
    # u Hilbertovom prostoru sa odgovarajucim brojem chestica
    ll_eigvec = np.zeros(n_dim)
    ind = np.nonzero(np.sum(fock_states, axis=1) == ll_eigval_ind+1)[0]
    for it, el in enumerate(ind):
        ll_eigvec[el] = ll_eigvec_trim[it]
    return ll_eigval, ll_eigvec


def plot_ntot_on_eps_t_graph(parameters=None, plt_density=21):
    if parameters is None:
        parameters = get_parameter()
    fock_states = get_fock_states(parameters)
    num_op = [get_number_operator(i) for i in range(parameters['L']**2)]
    num_op = np.array(num_op)

    eps_arr = np.linspace(-10, 0, plt_density)
    t_arr = np.linspace(-10, 0, plt_density)
    n_tot = np.zeros((eps_arr.shape[0], t_arr.shape[0]))

    for eps_it, eps in enumerate(eps_arr):
        for t_it, t in enumerate(t_arr):
            parameters['eps'] = eps
            parameters['t'] = t

            _, ll_eigvec = find_lowest_lying_eigenstate(parameters)
            first_ind = np.nonzero(ll_eigvec)[0][0]
            n_tot[eps_it, t_it] = np.sum(fock_states[first_ind])
            exp_sum = sum([calc_exp(num_op[i, :, :],
                           ll_eigvec) for i in range(parameters['L']**2)])
            if np.abs(n_tot[eps_it, t_it] - exp_sum) > 1e-6:
                print('Broj chestica nije dobar')
                print('eps =', eps, '\tt =', t)
                print('Iz Fokovog stanja:', n_tot[eps_it, t_it])
                print('Iz zbira ocekivanja:', exp_sum, '\n')

    title = '$N_{tot}$ u zavisnosti od $\\varepsilon$ i $t$, $V = ' + \
            str(parameters['V']) + '$'
    plt.title(title)
    plt.xlabel('$t$')
    plt.ylabel('$\\varepsilon$')
    plt.contourf(eps_arr, t_arr, n_tot)
    # plt.contourf(eps_arr, t_arr, n_tot, parameters['L']**2)
    plt.colorbar()
    plt.show()


def plot_exp_neighbors(parameters=None, V_max=101):
    if parameters is None:
        parameters = get_parameter()

    V_arr = np.linspace(0, V_max, V_max+1)
    exp_arr = np.zeros(V_arr.shape)
    for it, V in enumerate(V_arr):
        parameters['V'] = V
        _, ll_eigvec = find_lowest_lying_eigenstate(parameters)
        n0 = get_number_operator(0, parameters)
        n1 = get_number_operator(1, parameters)
        exp_arr[it] = calc_exp(n0 @ n1, ll_eigvec)

    plt.plot(V_arr, exp_arr)
    plt.xlabel('V')
    plt.ylabel('$\\langle n_i n_{i+1} \\rangle$')
    title = 'Verovatnoća nalaženja čestica na susednim poljima; ' + \
            '$\\varepsilon = ' + str(parameters['eps']) + '$ ' + \
            '$t = ' + str(parameters['t']) + '$'
    plt.title(title)
    plt.show()


def check_Wick_theorem(state_vec, parameters=None):
    if parameters is None:
        parameters = get_parameter()

    fock_states = get_fock_states()
    n_dim = fock_states.shape[0]
    hmltn = construct_single_particle_hamiltonian()

    cr_ops = np.empty((n_orb, n_dim, n_dim))
    an_ops = np.empty((n_orb, n_dim, n_dim))
    for i in range(n_orb):
        an_ops[i, :, :] = get_ac_operator(i, 'a')
        cr_ops[i, :, :] = get_ac_operator(i, 'c')

    ok = True
    print("Provera važenja Wickove teoreme:")
    print("i j lhs rhs")
    for i in range(n_orb):
        for j in range(n_orb):
            lhs_op = cr_ops[i] @ an_ops[i] @ cr_ops[j] @ an_ops[j]
            lhs_exp = calc_exp(lhs_op, state_vec)
            rhs_exp = calc_exp(cr_ops[i] @ an_ops[i], state_vec) * \
                calc_exp(cr_ops[j] @ an_ops[j], state_vec)
            rhs_exp += calc_exp(cr_ops[i] @ an_ops[j], state_vec) * \
                calc_exp(an_ops[i] @ cr_ops[j], state_vec)
            print(i, j, lhs_exp, rhs_exp)
            ok &= (lhs_exp == rhs_exp)
    if ok:
        print("Sve verovatnoće se poklapaju.")
    else:
        print("Postoje verovatnoće koje se ne poklapaju.")


def main(parameters=None):
    if parameters is None:
        parameters = get_parameter()

    single_ptcl_hmltn = construct_single_particle_hamiltonian(parameters)
    print('Jednochestichni Hamiltonijan:')
    print(single_ptcl_hmltn, end='\n\n')

    fock_states = get_fock_states(parameters)
    print('Fokova stanja:')
    print(fock_states, end='\n\n')

    # c0 = get_ac_operator(0, 'c', parameters)
    # print('Operator kreacije za prvu cesticu:')
    # print(c0)
    # a0 = get_ac_operator(0, 'a', parameters)
    # print('Operator anihilacije za prvu cesticu:')
    # print(a0, end='\n\n')

    op_hmltn = construct_hamiltonian_from_operators(parameters)
    # print('Hamiltonijan dobijen mnozenjem operatora:')
    # print(np.around(op_hmltn, 2), end='\n\n')

    splitovani = split_hamiltonian_into_blocks(parameters)
    for i, i_ptcl_hmltn in enumerate(splitovani):
        print('Broj chestica = ', i+1, ':\n', sep='')
        # print('Hamiltonijan:')
        # print(i_ptcl_hmltn, end='\n\n')
        evals, evecs = la.eigh(i_ptcl_hmltn)
        print('Eigenvrednosti:')
        print(evals, end='\n\n')
        print('Eigenvektori:')
        print(evecs, end='\n\n')

    ll_eigval, ll_eigvec = find_lowest_lying_eigenstate(parameters)
    print('Najniza svojstvena energija:', ll_eigval)
    print('Odgovarajuce svojstveno stanje:')
    print(ll_eigvec)
    # print(op_hmltn @ ket(ll_eigvec))
    # print(op_hmltn @ ket(ll_eigvec) - ll_eigval * ket(ll_eigvec))

    check_Wick_theorem(ll_eigvec)


if __name__ == '__main__':
    parameters = {'t': -2, 'eps': -3, 'V': 0, 'L': 2,
                  'max_occupancy': 1, 'statistic': 'Fermion'}
    parameters['n_orb'] = parameters['L'] ** 2
    n_orb = parameters['n_orb']

    ampl = 0
    noise = np.random.uniform(-ampl, ampl, parameters['n_orb'])
    parameters['noise'] = noise

    set_parameters(parameters)
    np.set_printoptions(precision=3, floatmode='maxprec', suppress=True)

    main()

    # fock_states = get_fock_states()
    # splt_hmltn = split_hamiltonian_into_blocks()
    # eigvals, eigvecs = [0] * n_orb, [0] * n_orb
    # for it, hmltn in enumerate(splt_hmltn):
    #     eigvals[it], eigvecs[it] = la.eigh(hmltn)
    # print(eigvals[0])
    # print(bra(eigvecs[0][0]) @ ket(eigvecs[0][2]))

    # ll_eigval, ll_eigvec = find_lowest_lying_eigenstate()
    # print(ll_eigval)
    # print(ll_eigvec)
