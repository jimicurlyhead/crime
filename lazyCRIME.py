#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from numpy import pi, sin, cos, sqrt, log, exp
import h5py
from scipy.interpolate import interp1d
from scipy.optimize import differential_evolution
from datetime import datetime
import sys

#requirements:
#   * shapes of strong and weak fields' waveforms are identical
#   * larger delay values correspond to the weak pulse arriving later

#define retrieval parameters:
file         = 'tiptoe_funkyRDW+NIR_nitrogen' #name of HDF5 file with raw data
n_om0        = 40 #number of frequency bands used for waveform retrieval
frac         = 0.998 #fraction of total fluence to be covered by frequency grid
IE0_eV       = 15.58 #ionisation energy (eV)
n_co         = 126 #number of available processors for multithreading

# =============================================================================
# PHYSICAL CONSTANTS:
# =============================================================================

#define relevant constants (CODATA):
e       = 1.602176634e-19 #elementary charge (C)
m_e     = 9.1093837015e-31 #electron mass (kg)
a_0     = 5.29177210903e-11 #Bohr radius (m)
h_bar   = 6.62607015e-34/(2*pi) #reduced Planck constant (J*s)
eps_0   = 8.8541878128e-12 #vacuum electric permittivity (F/m)
c_vac   = 2.99792458e8 #speed of light in vacuum (m/s)
mu_0    = 4*pi*1e-7 #vacuum magnetic permeability (N/A^2)

#derive further atomic units:
U_au    = h_bar**2 / (m_e * a_0**2) #atomic unit of energy (J)
t_au    = h_bar/U_au #atomic unit of time (s)
E_au    = U_au/(e*a_0) #atomic unit of electric field strength (V/m)
a_au    = (e*a_0)**2 / U_au #atomic unit of electric polarisability (C^2*m^2/J)
D2au    = 0.3934303 #1 Debye in atomic units

# =============================================================================
# LOAD EXPERIMENTAL DATA:
# =============================================================================

#set up name for snapshot file:
file_snp = file + '_nom={}_frac={}_IE={}eV'.format(n_om0, frac, IE0_eV)

#initialise wall clock time:
start = datetime.now()
now = datetime.now()

#load TIPTOE data:
with h5py.File(file + '.h5', 'r') as h5:
    # print('datasets in {}:'.format(file + '.h5'))
    # for key in h5:
    #     print('  ' + key)
    delay_fs = h5['delays (fs)'][:]
    trace = h5['signal N2+'][:]
    # trace = h5['rel. yield N2+'][:]
    wav_spec = h5['wavelengths (nm)'][:]
    spec = h5['spectral intensities (arb. u.)'][:]
    # wav_spec = h5['wavelengths weak pulse (nm)'][:]
    # spec = h5['spectral intensities weak pulse (arb. u.)'][:]

#centre delay frame:
delay_fs -= np.mean(delay_fs)
    
#convert experimental delay data to atomic units:
delay = delay_fs*1e-15/t_au

#define normalisation factor for deviation between model and experiment:
norm = np.sum((trace - 1)**2)

#define range for field-adaptive time grid:
Rt = max(delay) - min(delay)

#convert field-free ionisation energy to atomic units:
IE0 = IE0_eV*e/U_au

# =============================================================================
# FREQUENCY GRID:
# =============================================================================


def frac_indices(tr, frac):
    '''identifies the indices in the given trace that amount to the specified
    fractional cumulative sum
    
    tr   : one-dimensional distribution
    frac : fraction of the cumulative sum to be included [0; 1]'''
    
    #sort indices folloowing decending trace values and compute cumulative sum:
    j_sort = np.argsort(tr)[::-1]
    cs = np.cumsum(tr[j_sort])/np.sum(tr)
    
    #identify indices whose trace sum exceeds frational cumulative sum:
    j_cut = np.where(cs > frac)[0][0]
        
    return j_sort[:j_cut+1]


#convert spectrum from wavelength (nm) to circular frequency (at. u.):
om_spec = 2*pi*c_vac*t_au/(wav_spec*1e-9)
spec_om = spec * 2*pi*c_vac*t_au/om_spec**2

#sort spectrum with respect to ascending frequencies and compute increments:
j_sort = np.argsort(om_spec)
om_spec = om_spec[j_sort]
spec_om = spec_om[j_sort]
dom_spec = np.gradient(om_spec)

#identify wavelength regions that amount to majority of fluence:
j_maj = frac_indices(spec_om*dom_spec, frac)

#set up zero-centred circular frequency grid that covers whole spectrum:
om_a = min((om_spec - dom_spec/2)[j_maj])
om_b = max((om_spec + dom_spec/2)[j_maj])
Dom = np.sum(dom_spec[j_maj])/n_om0
j_om = np.arange(int(np.ceil(om_b/Dom)))
om = j_om * Dom
j_bom = np.arange(int(np.ceil(om_b/Dom))+1) - 0.5
bins_om = j_bom * Dom

#project input spectrum onto compressed circular frequency grid:
cs = np.cumsum(spec_om*dom_spec)
i_cs = interp1d(om_spec + dom_spec/2, cs, fill_value=0, bounds_error=False)
spec_om_maj = np.diff(i_cs(bins_om))/Dom

#compress circular frequency grid to region of fluence majority:
b_maj = np.zeros(len(om_spec))
b_maj[j_maj] = 1
b_sel = interp1d(om_spec, b_maj, kind='nearest', fill_value=0, bounds_error=False)(om)
j_om = np.where(b_sel == 1)[0]
om = j_om * Dom
n_om = len(om)
spec_om_maj = spec_om_maj[j_om]

#compute spectral amplitudes with normalised fluence:
amp = sqrt(spec_om_maj/(Dom*np.sum(spec_om_maj)))

#convert compressed spectrum back to wavelength (nm):
wav = 2*pi*c_vac*1e9*t_au/om
spec_wav_maj = (spec_om_maj*om**2)/(2*pi*c_vac*t_au)

#compute mean optical period:
Tm = 2*pi*np.sum(amp**2 / om)/np.sum(amp**2)

# =============================================================================
# LASER-ELECTRIC FIELD AND TUNNELLING RATE:
# =============================================================================


def efield_re(time, om_i, dom, amp_i, phi_i):
    '''composes the time domain representation of a laser-electric field in its
    real form based on its properties in the frequency domain. the spectral
    amplitudes and phases are assumed to be constant across the frequency bins.
    returns field in real form
    
    time  : time grid in atomic units, shape (n_t, n_tau)
    om_i  : circular frequency grid in atomic units, shape (n_om,)
    dom   : circular frequency increment in atomic units, scalar
    amp_i : spectral amplitudes in atomic units, shape (n_om,)
    phi_i : spectral phases in radians, shape (n_om,)'''
    
    #set up three-dimensional variables:
    t3 = time[..., None]
    ol3 = om_i[None, None] - dom/2
    oh3 = om_i[None, None] + dom/2
    a3 = amp_i[None, None]
    p3 = phi_i[None, None]
    
    #compose electric field in the time domain in real form:
    field_om = -a3 * (sin(p3 - oh3*t3) - sin(p3 - ol3*t3)) / t3
    field = np.sum(field_om, -1)
    
    return field


#prepare indices for Fourier-Frobenius companion matrix:
n_om_c = j_om[-1]
j_eye = np.arange(2*n_om_c - 1, dtype=int)
k_eye = np.arange(1, 2*n_om_c, dtype=int)
v_eye = np.ones(2*n_om_c-1)
j_bot = np.ones(2*n_om-1, dtype=int) * (2*n_om_c-1)
k_bot1 = n_om_c - j_om
k_bot2 = n_om_c + j_om[:-1]
k_bot = np.append(k_bot1, k_bot2)
j_tot = np.append(j_eye, j_bot)
k_tot = np.append(k_eye, k_bot)


def find_extrema(amp, phi):
    '''determines the time-domain extreme value positions of the given electric
    field from its frequency-domain properties by finding the eigenvalues of
    the underlying trigonometric polynomial's companion matrix. [1]
    the frequency grid of the polynomial is required to be equidistant and
    rooted at zero.
    
    [1] J. P. Boyd, J. Eng. Math. 56, 203--219 (2006)
    
    amp  : spectral amplitudes, shape (n_om,)
    phi  : spectral phases in radians, shape (n_om,)'''
    
    #set up Fourier-Frobenius companion matrix:
    a_j = amp * sin(phi) * j_om
    b_j = -amp * cos(phi) * j_om
    v_bot = -np.append(a_j + 1j*b_j, (a_j - 1j*b_j)[:-1]) / (a_j[-1] - 1j*b_j[-1])
    v_tot = np.append(v_eye, v_bot)
    A_jk = np.zeros((2*n_om_c, 2*n_om_c), dtype=complex)
    A_jk[(j_tot, k_tot)] = v_tot
    
    #compute eigenvalues of companion matrix and find extrema in time domain:
    eigen = np.linalg.eigvals(A_jk)
    t_ex = np.angle(eigen)/Dom
    e_ex = np.abs(np.abs(eigen) - 1)
    b_crit = e_ex < 1e-3 #discard extrema with large error
    t_ex = np.sort(t_ex[b_crit])
    
    return t_ex


def rate_adk(E_abs, IE):
    '''returns the tunnel ionisation rate in atomic units according to the ADK
    tunnel theory extended to the over-the-barrier regime [1], employing a
    value of alpha = 6 (hydrogen atom) for the empirical parameter in the
    barrier suppression extension. the initially bound electron is assumed to
    have no orbital angular momentum (l = 0), the initial system is assumed to
    be neutral (Zc = 0), and the amplitude of the bound electron wavefunction
    in the tunnelling region is assumed to be Cl = 2 [2].
    
    [1] X. M. Tong and C. D. Lin, J. Phys. B 38, 2593--2600 (2005)
    [2] X. M. Tong et al., Phys. Rev. A 66, 033402 (2002)
    
    E_abs : absolute electric field strength (atomic units)
    IE    : ionisation energy (atomic units)'''
            
    #get indices for absolute electric field strengths greater than zero:
    b_g0 = E_abs > 0
    
    #set up barrier parameter:
    k = sqrt(2*IE)
    
    #compute tunnelling rate:
    w1 = 2*(2*k**2/E_abs[b_g0])**(2/k - 1)
    w2 = exp(-2*k**3/(3*E_abs[b_g0]))
    w3 = exp(-12*E_abs[b_g0]/k**5)
    
    #compose tunnelling rate array:
    w = np.zeros(E_abs.shape)
    w[b_g0] = w1 * w2 * w3
    
    return w


# =============================================================================
# WAVEFORM RETRIEVAL:
# =============================================================================


def minfunc(para):
    'computes deviation between measured TIPTOE trace and model'
    
    #parse model parameters:
    delay0, F_hi, ratio_F = para[:3]
    phi = para[3:]
    
    #derive absolute amplitudes of strong and weak pulse:
    amp_hi = amp * sqrt(F_hi)
    amp_lo = amp_hi/sqrt(ratio_F)
    
    #compute offset delay grid:
    delay_o = delay - delay0
    
    #find extrema of strong field:
    t_hi = find_extrema(amp_hi, phi)
    
    #evaluate field extrema:
    E_hi = efield_re(t_hi[:, None], om, Dom, amp_hi, phi)[:, 0]
    
    #assess whether strong pulse triggers computable amount of ionisation:
    wi_hi = rate_adk(abs(E_hi), IE0)
    if 1 - np.prod(1 - wi_hi) > 0:
        
        #discard positive minima and negative maxima:
        d2E_c = cos(phi[None]) * cos(j_om[None]*Dom*t_hi[:, None])
        d2E_s = sin(phi[None]) * sin(j_om[None]*Dom*t_hi[:, None])
        d2E_hi = -np.sum(amp[None] * (d2E_c + d2E_s) * (j_om[None]*Dom)**2, 1)
        b_keep = E_hi*d2E_hi < 0
        t_hi = t_hi[b_keep]
        E_hi = E_hi[b_keep]
        
        #discard points at low relative field strength:
        b_keep = abs(E_hi) > 0.5*np.max(abs(E_hi))
        t_hi = t_hi[b_keep]

        #set up anchor points for trapezoid composite integration:
        n_trz = 7 #number of trapezoids per shoulder
        inc_trz = Tm/(6*n_trz)
        dt_anc = np.arange(-n_trz, n_trz+1) * inc_trz
        t_trz2 = t_hi[:, None] + dt_anc[None]

        #remove overlapping grid points:
        left = t_trz2[:, 0]
        right = t_trz2[:, -1]
        centre = (right[:-1] + left[1:])/2
        b_l = np.ones(t_trz2.shape, dtype=bool)
        b_l[:-1] = t_trz2[:-1] < centre[:, None] + inc_trz/2
        b_r = np.ones(t_trz2.shape, dtype=bool)
        b_r[1:] = t_trz2[1:] > centre[:, None] - inc_trz/2
        t_trz = np.sort(t_trz2[b_l & b_r])
    
        #evaluate electric fields on field-adapted grid:
        E_hi_trz = efield_re(t_trz[:, None], om, Dom, amp_hi, phi)
        E_lo_trz = efield_re(t_trz[:, None] - delay_o[None], om, Dom, amp_lo, phi)
        
        #compute instantaneous ionisation rates:
        wi_ref = rate_adk(abs(E_hi_trz[:, 0]), IE0)
        wi_sum = rate_adk(abs(E_hi_trz + E_lo_trz), IE0)
        
        #integrate along realtime axis to obtain cation populations:
        wm_ref = 0.5*(wi_ref[:-1] + wi_ref[1:])
        Pn_ref = exp(-np.sum(wm_ref*np.diff(t_trz)))
        wm_sum = 0.5*(wi_sum[:-1] + wi_sum[1:])
        Pn_sum = exp(-np.sum(wm_sum*np.diff(t_trz)[:, None], 0))
        Pc_ref = 1 - Pn_ref
        Pc_sum = 1 - Pn_sum
        
        #compute ionisation ratio and deviation between model and experiment:
        ratio_ion = Pc_sum/Pc_ref            
        chi = np.sum((trace - ratio_ion)**2)/norm
        
        #compute mean time of ionisation:
        t_m = np.sum(t_trz * wi_ref)/np.sum(wi_ref)
        zeta = abs(t_m)/(0.5*Rt)
        
        return chi + zeta
        
    else:
        
        #compute temporal standard deviation of intensity envelope:
        E_his = E_hi**2
        S_his = np.sum(E_his)
        t_hi_m = np.sum(E_his*t_hi)/S_his
        t_hi_ms = np.sum(E_his*t_hi**2)/S_his
        std = sqrt(t_hi_ms - t_hi_m**2)
        
        #compute intensity width relative to observation window:
        mu = std*2*sqrt(2*log(2))/Rt
        
        return max(mu, 1)
    
    
def show_fields(tau0, F_hi, ratio_F):
    '''plots laser-electric field of strong pulse for given set of parameters,
    assuming flat spectral phases'''
    
    #compute electric field:
    t = np.arange(-15e-15/t_au, 15e-15/t_au + 1, 1)
    phi = np.zeros(n_om)
    amp_hi = amp * sqrt(F_hi)
    amp_lo = amp_hi/sqrt(ratio_F)
    E_hi = efield_re(t, om, Dom, amp_hi, phi)[0]
    E_lo = efield_re(t, om, Dom, amp_lo, phi)[0]
    
    #plot fields:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(3, 2))
    plt.subplot2grid((2, 1), (0, 0))
    plt.plot(t*t_au*1e15, E_hi)
    plt.subplot2grid((2, 1), (1, 0))
    plt.plot(t*t_au*1e15, E_lo)
    plt.show()
    

def callback(xk, convergence):
    global now
    
    delta = (datetime.now() - now).total_seconds()
    if delta > 20:
                
        #save current parameter set:
        header = 'status     : running' + '\n'
        header += 'runtime    : {}'.format(datetime.now() - start) + '\n'
        header += 'minfunc(x) : {}'.format(minfunc(xk)) + '\n'
        para = '[{}'.format(xk[0])
        for j,x in enumerate(xk[1:]):
            if not (j+1)%3:
                sep = '\n'
            else:
                sep = ' '
            para += ',{}{}'.format(sep, x)
        para += ']\n'
        with open('{}.snp'.format(file_snp), 'w') as txt:
            txt.write(header + para)
        now = datetime.now()
        
        #update display of total optimisation target:
        ot = minfunc(xk)
        sys.stdout.write('\r')
        sys.stdout.write('   chi + zeta  : {}'.format(round(ot, 4)))
        sys.stdout.flush()
        now = datetime.now()


def extract_field(para0=None):
    '''extracts the laser-electric field from a TIPTOE trace employing the ADK
    tunnelling theory to describe the relative ionisation rate. the ionisation
    energy of the target is assumed to be field-invariant. optimises model to
    find spectral amplitude and phase at each point of the given circular-
    frequency grid
    
    para0 : initial guess for parameter vector'''
    
    #set up boundaries for parameter optimisation:
    bounds_tau0 = (-Rt/2, Rt/2)
    bounds_F = (1e-3, 1)
    bounds_rat = (100, 100000)
    bounds_phi = (0, 2*pi)
    bounds = [bounds_tau0] + [bounds_F] + [bounds_rat] + n_om*[bounds_phi]
    
    #find global optimum of electric field parameters:
    res = differential_evolution(minfunc, bounds, disp=False, x0=para0,
                                 updating='deferred', maxiter=250000,
                                 workers=n_co, callback=callback,
                                 polish=False, tol=1e-3)
    
    return res


if __name__ == "__main__":
    
    #extract electric field from TIPTOE trace:
    print('running waveform retrieval with {} processor(s)...'.format(n_co))
    print('   n_om        : {}'.format(n_om))
    print('   frac        : {}'.format(frac))
    print('   IE          : {} eV'.format(IE0_eV))
    print('   output file : {}.snp'.format(file_snp))
    result = extract_field()
    
    #save results:
    header = 'status     : finished' + '\n'
    header += 'runtime    : {}'.format(datetime.now() - start) + '\n'
    header += 'minfunc(x) : {}'.format(minfunc(result.x)) + '\n'
    para = '[{}'.format(result.x[0])
    for j,x in enumerate(result.x[1:]):
        if not (j+1)%3:
            sep = '\n'
        else:
            sep = ' '
        para += ',{}{}'.format(sep, x)
    para += ']\n'
    with open('{}.snp'.format(file_snp), 'w') as txt:
        txt.write(header + para)
        
    #update display of total optimisation target:
    ot = minfunc(result.x)
    sys.stdout.write('\r')
    sys.stdout.write('   chi + zeta  : {}'.format(round(ot, 4)))
    sys.stdout.flush()
    now = datetime.now()
    print('\n' + '   finished')
