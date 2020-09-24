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


def braket(vec1, vec2):
    exp_value = bra(vec1) @ ket(vec2)
    return np.float32(exp_value[0][0])


def calc_exp(operator, vec):
    return braket(vec, operator @ vec)


def get_xy_of_i(L, i):
    # za dati indeks chvora nadji koordinate
    return i % L, i//L


def get_i_of_xy(L, x, y):
    # za date periodichne koordinate nadji indeks
    x = x % L  # vrati x na opseg [0,L)
    y = y % L  # vrati y na opseg [0,L)
    return y*L + x


# NE RADI!!!!! Prekopiraj onaj Jaksin za 2d sistem
def draw_cluster(parameters):
    L = parameters['L']

    print('---', end='')
    for x in range(-1, L+1):
        print(x % L, '\t---', end=' ')
    print()


def construct_single_particle_hamiltonian(parameters):
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


def get_fock_states(parameters):
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


def get_ac_operator(i, a_or_c, parameters):
    # napravi operator kreacije/anihilacije na chvoru i
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


def get_state_ac_operator(state, a_or_c, parameters):
    n_orb = parameters['L'] ** 2
    n_dim = (parameters['max_occupancy'] + 1) ** (parameters['L'] ** 2)

    orb_ops = np.empty((n_orb, n_dim, n_dim))
    for it in range(n_orb):
        orb_ops[it, :, :] = get_ac_operator(it, a_or_c, parameters)

    state_op = np.zeros((n_dim, n_dim))
    for it in range(n_orb):
        state_op += state[it] * orb_ops[it, :, :]

    return state_op


def get_number_operator(i, parameters):
    a_op = get_ac_operator(i, 'a', parameters)
    c_op = get_ac_operator(i, 'c', parameters)

    return c_op @ a_op


def construct_hamiltonian_from_operators(parameters):
    # napravi many-body hamiltonian koristecci operatore kreacije i anihilacije
    L, V = parameters['L'], parameters['V']
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

                out += 0.5 * V * np.einsum('ij,jk,kl,lm->im',
                                           c[i], a[i], c[j], a[j])
        # print('Procenat izvrsenja DRUGOG fora je ' + str((i+1)*100/Norb))
    return out


def split_hamiltonian_into_blocks(parameters):
    n_orb = parameters['n_orb']

    fock_states = get_fock_states(parameters)
    many_body_hmltn = construct_hamiltonian_from_operators(parameters)
    out = [0] * (n_orb+1)
    for i in range(n_orb+1):
        ind = np.nonzero(np.sum(fock_states, axis=1) == i)[0]
        ind2d = tuple(np.meshgrid(ind, ind))
        out[i] = many_body_hmltn[ind2d]
    return out


def find_fock_eigenstates(parameters):
    n_orb = parameters['n_orb']
    n_dim = (parameters['max_occupancy'] + 1) ** n_orb
    fock_states = get_fock_states(parameters)

    split_hmltn = split_hamiltonian_into_blocks(parameters)
    eigvals, eigvecs = np.array([]), np.empty((0, n_dim))
    for it in range(n_orb+1):
        it_eigvals, it_eigvecs_trim = la.eigh(split_hmltn[it])
        it_eigvecs_trim = np.transpose(it_eigvecs_trim)
        eigvals = np.append(eigvals, it_eigvals)
        it_eigvecs = np.zeros((it_eigvecs_trim.shape[0], n_dim))
        for jt, el in enumerate(it_eigvecs_trim):
            it_eigvecs[jt][np.sum(fock_states, axis=1) == it] = el
        eigvecs = np.append(eigvecs, it_eigvecs, 0)

    # print(eigvals.shape, eigvecs.shape)
    return eigvals, eigvecs


def find_lowest_lying_eigenstate(parameters):
    eigvals, eigvecs = find_fock_eigenstates(parameters)
    ll_ind = np.argmin(eigvals)
    return eigvals[ll_ind], eigvecs[ll_ind]


def plot_ntot_on_eps_t_graph(parameters, plt_density=21):
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


def plot_exp_neighbors(parameters, V_max=101):
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


def check_Wick_theorem(state_vec, parameters):
    fock_states = get_fock_states(parameters)
    n_dim = fock_states.shape[0]
    hmltn = construct_single_particle_hamiltonian()

    cr_ops = np.empty((n_orb, n_dim, n_dim))
    an_ops = np.empty((n_orb, n_dim, n_dim))
    for i in range(n_orb):
        an_ops[i, :, :] = get_ac_operator(i, 'a', parameters)
        cr_ops[i, :, :] = get_ac_operator(i, 'c', parameters)

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


def constr_ground_state_from_operators(parameters):
    n_orb = parameters['n_orb']
    fock_states = get_fock_states(parameters)
    n_dim = fock_states.shape[0]

    eigvals, eigvecs = find_fock_eigenstates(parameters)
    single_ptcl = np.sum(fock_states, axis=1) == 1
    # prvo izvuchemo stanja koja odgovaraju jednochestichnom prostoru
    # a zatim izvucemo deo tih stanja koji odg jednochestichnom prostoru
    sptcl_eigvecs = eigvecs[single_ptcl][:, single_ptcl]
    sptcl_eigvals = eigvals[single_ptcl]

    ground_state = np.zeros(n_dim)
    ground_state[0] = 1
    for state in sptcl_eigvecs[sptcl_eigvals < 0]:
        cr_op = get_state_ac_operator(state, 'c', parameters)
        ground_state = cr_op @ ground_state

    return np.sum(sptcl_eigvals[sptcl_eigvals < 0]), ground_state


def calc_spectral_function(el_state, parameters):
    eigvals, eigvecs = find_fock_eigenstates(parameters)
    ground_energy, ground_state = \
        constr_ground_state_from_operators(parameters)
    step_log = -2          # log10 od razlike uzastopnih chlanova niza omega
    step = 10 ** step_log  # u ovom slucaju razlika uzastopnih je 0.01
    omega_max = 100  # spektralnu f racunamo na opsegu [-omega_max, omega_max]
    # broj tacaka takav da razlika izmedju uzastopnih bude tacno step
    no_points = 2 * omega_max * 10**(-step_log) + 1
    omega = np.linspace(-omega_max, omega_max, no_points)
    if (np.abs(omega[1] - omega[0] - step) > 1e-10):
        raise ValueError('los korak')
    func = np.zeros(omega.shape)

    el_an = get_state_ac_operator(el_state, 'a', parameters)
    el_cr = get_state_ac_operator(el_state, 'c', parameters)
    an_state = el_an @ ground_state
    cr_state = el_cr @ ground_state
    for it in range(len(eigvals)):
        an_ampl = np.abs(braket(eigvecs[it], an_state)) ** 2
        cr_ampl = np.abs(braket(eigvecs[it], cr_state)) ** 2
        en_diff = ground_energy - eigvals[it]
        # zaokruzujemo en_diff na tacnost koju ima i omega
        en_diff = np.around(en_diff, -step_log)
        omega_ind = np.argwhere(np.abs(omega - en_diff) < step/10)[0]
        if len(omega_ind) != 1:
            print('greska')
            print(en_diff)
        func[omega_ind[0]] += np.around(cr_ampl + an_ampl, 4)

    return omega, func


def total_spectral_function(parameters):
    eigvals, eigvecs = find_fock_eigenstates(parameters)
    single_ptcl = np.sum(fock_states, axis=1) == 1
    sptcl_eigvals = eigvals[single_ptcl]
    sptcl_eigvecs = eigvecs[single_ptcl][:, single_ptcl]
    omega = np.linspace(-100, 100, 20001)
    br_func = len(sptcl_eigvals)
    spec_func = np.zeros(omega.shape)
    for i in range(br_func):
        spec_func += calc_spectral_function(sptcl_eigvecs[i], parameters)[1]
    return omega, 1/br_func * spec_func

def step_function(x, parameters):
    if x>=0:
        return 1
    else:
        return 0

def green_function(alpha, beta, parameters):
    step_log = -2          # log10 od razlike uzastopnih chlanova niza omega
    step = 10 ** step_log  # u ovom slucaju razlika uzastopnih je 0.01
    g_max = 100  # spektralnu f racunamo na opsegu [-omega_max, omega_max]
    # broj tacaka takav da razlika izmedju uzastopnih bude tacno step
    no_points = 2 * g_max * 10**(-step_log) + 1
    p = g = np.linspace(-g_max, g_max, no_points)
    ground_energy, ground_state=constr_ground_state_from_operators(parameters)
    for i in range(len(g)):
        g[i]=-1j*step_function(i, parameters)*(braket(bra(ground_state)*np.exp(1j*i*ground_energy)*get_state_ac_operator(alpha,'a', parameters),np.exp(-1j*i*ground_energy)*get_state_ac_operator(beta, 'c', parameters)*ket(ground_state)) + braket(bra(ground_state)*get_state_ac_operator(beta,'c', parameters)*np.exp(1j*i*ground_energy),get_state_ac_operator(alpha, 'a', parameters)*np.exp(-1j*i*ground_energy)*ket(ground_state)))
    return p, g

def main(parameters):
    print('Parametri:', parameters)

    fock_states = get_fock_states(parameters)
    print('Fokova stanja:')
    print(fock_states, end='\n\n')

    op_hmltn = construct_hamiltonian_from_operators(parameters)
    print('Hamiltonijan dobijen mnozenjem operatora:')
    print(np.around(op_hmltn, 2), end='\n\n')

    eigvals, eigvecs = find_fock_eigenstates(parameters)
    print('Svojstvene energije:', eigvals)
    print('Svojstvena stanja:', eigvecs, sep='\n', end='\n\n')

    ll_eigval, ll_eigvec = constr_ground_state_from_operators(parameters)
    print('Najniza svojstvena energija:', ll_eigval)
    print('Odgovarajuce svojstveno stanje:')
    print(ll_eigvec)

    # check_Wick_theorem(ll_eigvec)

    omega, func = calc_spectral_function([1, 0, 0, 0], parameters)
    plt.plot(omega, func)
    plt.show()


if __name__ == '__main__':
    parameters = {'t': -2, 'eps': -3, 'V': 10, 'L': 2,
                  'max_occupancy': 1, 'statistic': 'Fermion'}
    parameters['n_orb'] = parameters['L'] ** 2
    n_orb = parameters['n_orb']

    ampl = 0
    noise = np.random.uniform(-ampl, ampl, parameters['n_orb'])
    parameters['noise'] = noise

    set_parameters(parameters)
    np.set_printoptions(precision=3, floatmode='maxprec', suppress=True)

    main(parameters)
'''
    fock_states = get_fock_states(parameters)
    eigvals, eigvecs = find_fock_eigenstates(parameters)
    single_ptcl = np.sum(fock_states, axis=1) == 1
    sptcl_eigvals = eigvals[single_ptcl]
    sptcl_eigvecs = eigvecs[single_ptcl][:, single_ptcl]
    omega, func = total_spectral_function(parameters)
    print('svojstvene energije jednochestichnog hamiltonijana:', sptcl_eigvals)
    print('pikovi spektralne funkcije:', omega[func != 0])
    print('norma:', np.sum(func))
    plt.plot(omega, func)
    plt.show()
'''
plt.plot(green_function([1,0,0,0], [1,0,0,0], parameters))
plt.show()
plt.plot(green_function([0,1,0,0], [0,1,0,0], parameters))
plt.show()
plt.plot(green_function([0,0,1,0], [0,0,1,0], parameters))
plt.show()
plt.plot(green_function([0,0,0,1], [0,0,0,1], parameters))
plt.show()