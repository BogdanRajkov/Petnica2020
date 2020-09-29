import numpy as np
import numpy.linalg as la
from numpy.random import uniform
import itertools
import matplotlib.pyplot as plt


class SystemData:
    def __init__(self, parameters):
        """
        initialise system with some basic info contained in a dictionary
        example:
        parameters = {'t': -2, 'eps': -3, 'V': 10, 'L': 2,
                      'max_occupancy': 2, 'statistic': 'Boson',
                      'noise_ampl': 1e-3}
        """

        # podaci koji moraju biti uneti
        self.L = parameters['L']
        self.statistic = parameters['statistic']
        if self.statistic not in ['Boson', 'Fermion']:
            raise ValueError("unknown statistic")
        if self.statistic == 'Fermion':
            self.max_occupancy = 1
        else:
            self.max_occupancy = parameters['max_occupancy']
        self.t, self.eps, self.V = \
            parameters['t'], parameters['eps'], parameters['V']
        try:
            self.noise_ampl = basic_info['noise_ampl']
        except KeyError:
            self.noise_ampl = 0

        # izvedeni podaci
        self.n_orb = self.L ** 2
        self.n_dim = (self.max_occupancy + 1) ** self.n_orb
        self.noise = uniform(-self.noise_ampl, self.noise_ampl, self.n_orb)

    def get_basic_info(self):
        """
        returns a dictionary which can later be used to initialise other
        instances of the class, perhaps with some parameters modified
        """

        return {'statistic': self.statistic, 'noise_ampl': self.noise_ampl,
                'L': self.L, 'eps': self.eps, 't': self.t, 'V': self.V}


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
    L = parameters.L

    print('---', end='')
    for x in range(-1, L+1):
        print(x % L, '\t---', end=' ')
    print()


def construct_single_particle_hamiltonian(parameters):
    try:
        return parameters.sptcl_hamiltonian
    except AttributeError:
        pass

    L = parameters.L
    n_orb = parameters.n_orb
    out = np.zeros((n_orb, n_orb))

    for i in range(n_orb):
        out[i, i] = parameters.eps + parameters.noise[i]
        x, y = get_xy_of_i(L, i)
        for d in [-1, 1]:  # najblizhi susedi...
            for a, b in [(x, y+d), (x+d, y)]:  # po x i y osi
                j = get_i_of_xy(L, a, b)  # nadji indeks suseda
                out[i, j] = out[j, i] = parameters.t  # popuni chlan

    parameters.sptcl_hamiltonian = out
    return parameters.sptcl_hamiltonian


def get_fock_states(parameters):
    try:
        return parameters.fock_states
    except AttributeError:
        pass

    fock_states = list(
        itertools.product(range(parameters.max_occupancy + 1),
                          repeat=parameters.L**2))
    # sortiraj stanja po ukupnom broju chestica
    fock_states = sorted(fock_states, key=lambda x: sum(x))
    # pretvorimo u numpy.array
    for fsi, fs in enumerate(fock_states):
        fock_states[fsi] = np.array(list(fs))

    parameters.fock_states = np.array(fock_states)
    return parameters.fock_states


def get_ac_operator(i, a_or_c, parameters):
    """napravi operator kreacije/anihilacije na chvoru i"""
    try:
        if a_or_c == 'a':
            return parameters.an_ops[i, :, :]
        elif a_or_c == 'c':
            return parameters.cr_ops[i, :, :]
        else:  # ako nije ni 'a' ni 'c' onda je u pitanju greska
            raise ValueError('unknown type of operator')
    except AttributeError:
        pass
    n_orb, n_dim, max_occupancy = \
        parameters.n_orb, parameters.n_dim, parameters.max_occupancy

    if a_or_c == 'c':
        shift = 1
    elif a_or_c == 'a':
        shift = -1

    fock_states = get_fock_states(parameters)
    out = np.zeros((n_orb, n_dim, n_dim))
    for it in range(n_orb):
        for fs1i, fs1 in enumerate(fock_states):
            fs2 = np.array(fs1)
            fs2[it] += shift
            if fs2[it] not in range(max_occupancy+1):
                continue  # preskachemo ako ispadne iz opsega
            fs2i = np.nonzero((fock_states == fs2).all(axis=1))[0][0]

            if parameters.statistic == 'Fermion':
                sgn = (-1)**(sum(fs2[it+1:]))
            elif parameters.statistic == 'Boson':
                raise NotImplementedError('kod za bozone nije popravljen')
            out[it, fs2i, fs1i] = sgn

    if a_or_c == 'a':
        parameters.an_ops = out
        return parameters.an_ops[i, :, :]
    elif a_or_c == 'c':
        parameters.cr_ops = out
        return parameters.cr_ops[i, :, :]


def get_state_ac_operator(state, a_or_c, parameters):
    n_orb, n_dim = parameters.n_orb, parameters.n_dim

    # ovo je samo glup nacin da se osigura postojanje
    # odgovarajucih operatora u parameters
    get_ac_operator(0, a_or_c, parameters)
    if a_or_c == 'a':
        orb_ops = parameters.an_ops
    elif a_or_c == 'c':
        orb_ops = parameters.cr_ops

    state_op = np.zeros((n_dim, n_dim))
    for it in range(n_orb):
        state_op += state[it] * orb_ops[it, :, :]

    return state_op


def get_number_operator(i, parameters):
    a_op = get_ac_operator(i, 'a', parameters)
    c_op = get_ac_operator(i, 'c', parameters)

    return c_op @ a_op


def construct_hamiltonian_from_operators(parameters):
    """napravi many-body hamiltonian koristecci
    operatore kreacije i anihilacije"""
    try:
        return parameters.hamiltonian
    except AttributeError:
        pass

    L, eps, t, V, n_orb, n_dim, noise = \
        parameters.L, parameters.eps, parameters.t, parameters.V, \
        parameters.n_orb, parameters.n_dim, parameters.noise
    fock_states = get_fock_states(parameters)
    out = np.zeros((n_dim, n_dim))
    # glup nacin da se osigura postojanje an i cr operatora
    get_ac_operator(0, 'a', parameters)
    get_ac_operator(0, 'c', parameters)
    a, c = parameters.an_ops, parameters.cr_ops

    for i in range(n_orb):
        out += (eps + noise[i]) * c[i] @ a[i]
        x, y = get_xy_of_i(L, i)
        # najblizhi susedi, ali pazimo na
        # klasteru 2x2 da ne rachunamo dvaput isto
        for d in ([-1, 1] if L > 2 else [1]):
            for it1, b in [(x, y+d), (x+d, y)]:  # po x i y osi
                j = get_i_of_xy(L, it1, b)  # nadji indeks suseda
                out += t * c[i] @ a[j]

                out += 0.5 * V * np.einsum('ij,jk,kl,lm->im',
                                           c[i], a[i], c[j], a[j])
        # print('Procenat izvrsenja DRUGOG fora je ' + str((i+1)*100/n_orb))

    parameters.hamiltonian = out
    return parameters.hamiltonian


def split_hamiltonian_into_blocks(parameters):
    try:
        return parameters.split_hamiltonian
    except AttributeError:
        pass

    n_orb = parameters.n_orb
    fock_states = get_fock_states(parameters)
    many_body_hmltn = construct_hamiltonian_from_operators(parameters)
    out = [0] * (n_orb+1)
    for i in range(n_orb+1):
        ind = np.nonzero(np.sum(fock_states, axis=1) == i)[0]
        ind2d = tuple(np.meshgrid(ind, ind))
        out[i] = many_body_hmltn[ind2d]

    parameters.split_hamiltonian = out
    return parameters.split_hamiltonian


def find_fock_eigenstates(parameters):
    try:
        return parameters.fock_eigvals, parameters.fock_eigvecs
    except AttributeError:
        pass

    n_orb, n_dim = parameters.n_orb, parameters.n_dim
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

    parameters.fock_eigvals, parameters.fock_eigvecs = eigvals, eigvecs
    return parameters.fock_eigvals, parameters.fock_eigvecs


def find_lowest_lying_eigenstate(parameters):
    try:
        return parameters.ground_energy, parameters.ground_state
    except AttributeError:
        pass

    eigvals, eigvecs = find_fock_eigenstates(parameters)
    ll_ind = np.argmin(eigvals)

    parameters.ground_energy = eigvals[ll_ind]
    parameters.ground_state = eigvecs[ll_ind]
    return parameters.ground_energy, parameters.ground_state


def constr_ground_state_from_operators(parameters):
    n_orb, n_dim = parameters.n_orb, parameters.n_dim
    fock_states = get_fock_states(parameters)

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

    ground_energy = np.sum(sptcl_eigvals[sptcl_eigvals < 0])
    return ground_energy, ground_state


def plot_ntot_on_eps_t_graph(parameters, plt_density=21):
    V, n_orb, n_dim = parameters.V, parameters.n_orb, parameters.n_dim
    fock_states = get_fock_states(parameters)
    num_op = np.empty((n_orb, n_dim, n_dim))
    for it in range(n_orb):
        num_op[it, :, :] = get_number_operator(it, parameters)

    eps_arr = np.linspace(-10, 0, plt_density)
    t_arr = np.linspace(-10, 0, plt_density)
    n_tot = np.zeros((eps_arr.shape[0], t_arr.shape[0]))

    basic_info = parameters.get_basic_info()
    for eps_it, eps in enumerate(eps_arr):
        for t_it, t in enumerate(t_arr):
            basic_info['eps'], basic_info['t'] = eps, t
            temp_param = SystemData(basic_info)

            _, ll_eigvec = find_lowest_lying_eigenstate(temp_param)
            first_ind = np.nonzero(ll_eigvec)[0][0]
            n_tot[eps_it, t_it] = np.sum(fock_states[first_ind])
            # n_tot[eps_it, t_it] = np.sum(fock_states[first_ind, :])
            exp_sum = sum([calc_exp(num_op[i, :, :],
                           ll_eigvec) for i in range(n_orb)])
            if np.abs(n_tot[eps_it, t_it] - exp_sum) > 1e-6:
                print('Broj chestica nije dobar')
                print('eps =', eps, '\tt =', t)
                print('Iz Fokovog stanja:', n_tot[eps_it, t_it])
                print('Iz zbira ocekivanja:', exp_sum, '\n')

    title = '$N_{tot}$ u zavisnosti od $\\epsilon$ i $t$, $V = ' + str(V) + '$'
    plt.title(title)
    plt.xlabel('$t$')
    plt.ylabel('$\\epsilon$')
    plt.contourf(eps_arr, t_arr, n_tot)
    plt.colorbar()
    plt.show()


def plot_exp_neighbors(parameters, V_max=101):
    n_orb, n_dim = parameters.n_orb, parameters.n_dim
    V_arr = np.linspace(0, V_max, V_max+1)
    exp_arr = np.zeros(V_arr.shape)
    num_op = np.empty((n_orb, n_dim, n_dim))
    for it in range(n_orb):
        num_op[it, :, :] = get_number_operator(it, parameters)

    basic_info = parameters.get_basic_info()
    for it, V in enumerate(V_arr):
        basic_info['V'] = V
        temp_param = SystemData(basic_info)
        ll_eigval, ll_eigvec = find_lowest_lying_eigenstate(temp_param)
        exp_sum = sum(
            [calc_exp(num_op[i, :, :], ll_eigvec) for i in range(n_orb)])
        exp_arr[it] = calc_exp(num_op[0, :, :] @ num_op[1, :, :], ll_eigvec)

        # print('V = {:4d}, gr_energy = {:4.2f}, n_tot = {:4.1f}'.
        #       format(int(V), ll_eigval, exp_sum))

    plt.plot(V_arr, exp_arr)
    plt.xlabel('V')
    plt.ylabel('$\\langle n_i n_{i+1} \\rangle$')
    title = 'Verovatnoća nalaženja čestica na susednim poljima; ' + \
            '$\\varepsilon = ' + str(parameters.eps) + '$ ' + \
            '$t = ' + str(parameters.t) + '$'
    plt.title(title)
    plt.show()


def check_Wick_theorem(state_vec, parameters):
    n_orb, n_dim = parameters.n_orb, parameters.n_dim
    fock_states = get_fock_states(parameters)
    hmltn = construct_single_particle_hamiltonian(parameters)

    get_ac_operator(0, 'a', parameters)
    get_ac_operator(0, 'c', parameters)
    an_ops, cr_ops = parameters.an_ops, parameters.cr_ops

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
    fock_states = get_fock_states(parameters)
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


def plot_total_spectral_function(parameters):
    omega, func = total_spectral_function(parameters)
    print('pikovi spektralne funkcije:', omega[func != 0])

    plt.title('Totalna spektralna funkcija')
    plt.xlabel('$\\omega$')
    plt.ylabel('$\\rho(\\omega)$')
    plt.plot(omega, func)
    plt.show()


def step_function(x, parameters):
    if x >= 0:
        return 1
    else:
        return 0


def constr_time_propagator(t, parameters):
    eigvals, eigvecs = find_fock_eigenstates(parameters)
    n_dim = len(eigvals)
    site_to_eigenstate = np.empty((n_dim, n_dim), dtype=np.complex64)
    eigenstate_to_site = np.empty((n_dim, n_dim), dtype=np.complex64)

    time_prop = np.zeros((n_dim, n_dim), dtype=np.complex64)
    di = np.diag_indices(n_dim)
    time_prop[di] = np.exp(1j*t*eigvals)

    for it in range(n_dim):
        site_to_eigenstate[:, it] = eigvecs[it]
    site_vec = np.zeros(n_dim)
    for it in range(n_dim):
        site_vec[it] += 1
        eigenstate_to_site[:, it] = site_to_eigenstate @ site_vec
        site_vec[it] -= 1

    return eigenstate_to_site @ time_prop @ site_to_eigenstate


def green_function(alpha, beta, parameters):
    step_log = -2          # log10 od razlike uzastopnih chlanova niza omega
    step = 10 ** step_log  # u ovom slucaju razlika uzastopnih je 0.01
    g_max = 100  # spektralnu f racunamo na opsegu [-omega_max, omega_max]
    # broj tacaka takav da razlika izmedju uzastopnih bude tacno step
    no_points = 2 * g_max * 10**(-step_log) + 1
    p = np.linspace(-g_max, g_max, no_points)
    g = np.zeros_like(p)
    ground_energy, ground_state = \
        constr_ground_state_from_operators(parameters)
    ground_bra = np.array(bra(ground_state), dtype=np.complex64)
    ground_ket = np.array(ket(ground_state), dtype=np.complex64)
    an_alpha = get_state_ac_operator(alpha, 'a', parameters)
    an_alpha = np.array(an_alpha, dtype=np.complex64)
    cr_beta = get_state_ac_operator(beta, 'c', parameters)
    cr_beta = np.array(cr_beta, dtype=np.complex64)
    for it, t in enumerate(p):
        time_prop_forw = constr_time_propagator(t, parameters)
        time_prop_back = constr_time_propagator(-t, parameters)
        g[it] = -1j * step_function(t, parameters) * \
            ((ground_bra @ time_prop_forw @ an_alpha @ time_prop_back @
              cr_beta @ ground_ket)[0][0]
             + (ground_bra @ cr_beta @ time_prop_forw @ an_alpha @
                time_prop_back @ ground_ket)[0][0])
    return p, g


def main(parameters):
    print('Osnovni podaci o sistemu:')
    print('\tVelicina osnovne jedinice: {0}x{0}'.format(parameters.L))
    print('\tStatistika:', parameters.statistic)
    print('\tAmplituda suma:', parameters.noise_ampl)
    print('\teps, t, V = {0}, {1}, {2}'.format(parameters.eps, parameters.t,
                                               parameters.V))

    fock_states = get_fock_states(parameters)
    print('Fokova stanja:')
    print(fock_states, end='\n\n')

    op_hmltn = construct_hamiltonian_from_operators(parameters)
    print('Hamiltonijan dobijen mnozenjem operatora:')
    print(np.around(op_hmltn, 2), end='\n\n')

    eigvals, eigvecs = find_fock_eigenstates(parameters)
    print('Svojstvene energije:', eigvals)
    print('Svojstvena stanja:', eigvecs, sep='\n', end='\n\n')

    ll_eigval, ll_eigvec = find_lowest_lying_eigenstate(parameters)
    print('Najniza svojstvena energija:', ll_eigval)
    print('Odgovarajuce svojstveno stanje:')
    print(ll_eigvec)

    # plot_ntot_on_eps_t_graph(parameters)
    # plot_exp_neighbors(parameters)
    # check_Wick_theorem(ll_eigvec, parameters)
    # plot_total_spectral_function(parameters)


if __name__ == '__main__':
    basic_info = {'eps': -8, 't': -2, 'V': 2, 'L': 2,
                  'statistic': 'Fermion', 'noise_ampl': 1e-3}
    parameters = SystemData(basic_info)
    np.set_printoptions(precision=3, floatmode='maxprec', suppress=True)

    # main(parameters)
    i = np.array([1, 0, 0, 0])
    p, g = green_function(i, i, parameters)
    plt.plot(p, g)
    plt.show()
