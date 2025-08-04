#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import h5py
import numpy as np
from numpy import pi, sin, cos, sqrt, exp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
figfac = 0.6 #figure scaling factor
plt.rcParams.update({'font.size': 10*figfac})

#requirements:
#   * larger delay values correspond to the weak pulse arriving later

#define time identifier of measurement:
time_id     = '240416_1458_2'

#define retrieval parameters:
species     = 'Ne'
frac        = 0.998 #fraction of total fluence to be covered by frequency grid
q_lo        = 0.95 #minimum fraction of weak pulse's fluence within delay frame
n_om0       = 20
check_input = False #boolean, sets all spectral phases to 0

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

#set up dictionary of field-free vertical ionisation energies in eV:
d_IE0_eV = {'He':24.587, 'Ne':21.565, 'Ar':15.759, 'N2':15.58, 'H2O':12.600}

# =============================================================================
# LOAD EXPERIMENTAL DATA:
# =============================================================================

#set up name for snapshot file:
file_snp = 'crime_{}_{}_nom={}_frac={}_qlo={}.snp'.format(time_id, species, n_om0, frac, q_lo)

#load TIPTOE data:
file = 'tiptoe_{}.h5'.format(time_id)
with h5py.File(file, 'r') as h5:
    print('datasets in {}:'.format(file))
    for key in h5:
        print('  ' + key)
    delay_fs = h5['target delays (fs)'][:]
    trace = h5['signal {}+'.format(species)][:]
    wav_spec_hi = h5['wavelengths strong pulse (nm)'][:]
    wav_spec_lo = h5['wavelengths weak pulse (nm)'][:]
    spec_hi = h5['spectral intensities strong pulse (arb. u.)'][:]
    spec_lo = h5['spectral intensities weak pulse (arb. u.)'][:]
    atts = dict(h5.attrs)
    F_hi = atts['peak fluence strong pulse (J/m^2)']
    F_lo = atts['peak fluence weak pulse (J/m^2)']
    
#centre delay frame:
delay_fs -= np.mean(delay_fs)
    
#convert experimental delay data to atomic units:
delay = delay_fs*1e-15/t_au

#define normalisation factor for deviation between model and experiment:
norm = np.sum((trace - 1)**2)

#define range for field-adaptive time grid:
Rt = max(delay) - min(delay)

#convert field-free ionisation energy to atomic units:
IE0 = d_IE0_eV[species]*e/U_au

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
om_spec_hi = 2*pi*c_vac*t_au/(wav_spec_hi*1e-9)
spec_om_hi = spec_hi * 2*pi*c_vac*t_au/om_spec_hi**2
om_spec_lo = 2*pi*c_vac*t_au/(wav_spec_lo*1e-9)
spec_om_lo = spec_lo * 2*pi*c_vac*t_au/om_spec_lo**2

#sort spectrum with respect to ascending frequencies and compute increments:
j_sort_hi = np.argsort(om_spec_hi)
om_spec_hi = om_spec_hi[j_sort_hi]
spec_om_hi = spec_om_hi[j_sort_hi]
dom_spec_hi = np.gradient(om_spec_hi)
j_sort_lo = np.argsort(om_spec_lo)
om_spec_lo = om_spec_lo[j_sort_lo]
spec_om_lo = spec_om_lo[j_sort_lo]
dom_spec_lo = np.gradient(om_spec_lo)

#compute time-integrated intensities from peak fluences (at. u.):
Esq_f_hi = 2*F_hi/(c_vac*eps_0) / (t_au*E_au**2)
Esq_f_lo = 2*F_lo/(c_vac*eps_0) / (t_au*E_au**2)

#compute time-integrated intensities from spectral intensities (arb. u.):
Dom_spec_hi = np.gradient(om_spec_hi)
Dom_spec_lo = np.gradient(om_spec_lo)
Esq_i_hi = 2*pi * np.sum(spec_om_hi*Dom_spec_hi)
Esq_i_lo = 2*pi * np.sum(spec_om_lo*Dom_spec_lo)

#bring spectral intensities to absolute scale:
spec_om_hi *= Esq_f_hi/Esq_i_hi
spec_om_lo *= Esq_f_lo/Esq_i_lo

#identify wavelength regions that amount to majority of fluence:
j_maj_hi = frac_indices(spec_om_hi*dom_spec_hi, frac)
j_maj_lo = frac_indices(spec_om_lo*dom_spec_lo, frac)

#set up zero-centred circular frequency grid that covers whole spectrum:
om_a_hi = min((om_spec_hi - dom_spec_hi/2)[j_maj_hi])
om_b_hi = max((om_spec_hi + dom_spec_hi/2)[j_maj_hi])
Dom_hi = np.sum(dom_spec_hi[j_maj_hi])/n_om0
j_om_hi = np.arange(int(np.ceil(om_b_hi/Dom_hi)))
om_hi = j_om_hi * Dom_hi
j_bom_hi = np.arange(int(np.ceil(om_b_hi/Dom_hi))+1) - 0.5
bins_om_hi = j_bom_hi * Dom_hi
om_a_lo = min((om_spec_lo - dom_spec_lo/2)[j_maj_lo])
om_b_lo = max((om_spec_lo + dom_spec_lo/2)[j_maj_lo])
Dom_lo = np.sum(dom_spec_lo[j_maj_lo])/n_om0
j_om_lo = np.arange(int(np.ceil(om_b_lo/Dom_lo)))
om_lo = j_om_lo * Dom_lo
j_bom_lo = np.arange(int(np.ceil(om_b_lo/Dom_lo))+1) - 0.5
bins_om_lo = j_bom_lo * Dom_lo

#project input spectrum onto compressed circular frequency grid:
cs_hi = np.cumsum(spec_om_hi*dom_spec_hi)
i_cs_hi = interp1d(om_spec_hi + dom_spec_hi/2, cs_hi, fill_value=0, bounds_error=False)
spec_om_maj_hi = np.diff(i_cs_hi(bins_om_hi))/Dom_hi
cs_lo = np.cumsum(spec_om_lo*dom_spec_lo)
i_cs_lo = interp1d(om_spec_lo + dom_spec_lo/2, cs_lo, fill_value=0, bounds_error=False)
spec_om_maj_lo = np.diff(i_cs_lo(bins_om_lo))/Dom_lo

#compress circular frequency grid to region of fluence majority:
b_maj_hi = np.zeros(len(om_spec_hi))
b_maj_hi[j_maj_hi] = 1
b_sel_hi = interp1d(om_spec_hi, b_maj_hi, kind='nearest', fill_value=0, bounds_error=False)(om_hi)
j_om_hi = np.where(b_sel_hi == 1)[0]
om_hi = j_om_hi * Dom_hi
n_om_hi = len(om_hi)
spec_om_maj_hi = spec_om_maj_hi[j_om_hi]
b_maj_lo = np.zeros(len(om_spec_lo))
b_maj_lo[j_maj_lo] = 1
b_sel_lo = interp1d(om_spec_lo, b_maj_lo, kind='nearest', fill_value=0, bounds_error=False)(om_lo)
j_om_lo = np.where(b_sel_lo == 1)[0]
om_lo = j_om_lo * Dom_lo
n_om_lo = len(om_lo)
spec_om_maj_lo = spec_om_maj_lo[j_om_lo]

#compute spectral amplitudes on new spectral grids:
amp_hi = sqrt(spec_om_maj_hi)
amp_lo = sqrt(spec_om_maj_lo)

#compute mean optical period:
Tm_hi = 2*pi*np.sum(amp_hi**2 / om_hi)/np.sum(amp_hi**2)

#prepare time-integrated intensities:
It_hi = 2*pi*np.sum(amp_hi**2)*Dom_hi
It_lo = 2*pi*np.sum(amp_lo**2)*Dom_lo
II_hi = np.sum(amp_hi**2)
II_lo = np.sum(amp_lo**2)

# =============================================================================
# LASER-ELECTRIC FIELD AND TUNNELLING RATE:
# =============================================================================


def efield_c(time, om_i, Dom, amp_i, phi_i):
    '''composes the time domain representation of a laser-electric field in its
    complex form based on its properties in the frequency domain. the spectral
    amplitudes and phases are assumed to be constant in between the given
    frequency bins bom_i.
    returns field in complex form
    
    time  : time grid in atomic units, shape (n_t, n_tau)
    om_i  : circular frequency grid in ascending order, shape (n_om,)
    Dom   : increment of circular frequency grid, scalar
    amp_i : spectral amplitudes in atomic units, shape (n_om,)
    phi_i : spectral phases in radians, shape (n_om,)'''
    
    #set up three-dimensional variables:
    t3 = time[..., None]
    ol3 = om_i[None, None] - Dom/2
    oh3 = om_i[None, None] + Dom/2
    a3 = amp_i[None, None]
    p3 = phi_i[None, None]
    
    #compose electric field in the time domain in complex form:
    field_om = a3*1j * (exp(1j*(p3 - oh3*t3)) - exp(1j*(p3 - ol3*t3))) / t3
    field = np.sum(field_om, -1)
    
    return field


def efield_re(time, om_i, Dom, amp_i, phi_i):
    '''composes the time domain representation of a laser-electric field in its
    real form based on its properties in the frequency domain. the spectral
    amplitudes and phases are assumed to be constant in between the given
    frequency bins bom_i.
    returns field in real form
    
    time  : time grid in atomic units, shape (n_t, n_tau)
    om_i  : circular frequency grid in ascending order, shape (n_om,)
    Dom   : increment of circular frequency grid, scalar
    amp_i : spectral amplitudes in atomic units, shape (n_om,)
    phi_i : spectral phases in radians, shape (n_om,)'''
    
    #set up three-dimensional variables:
    t3 = time[..., None]
    ol3 = om_i[None, None] - Dom/2
    oh3 = om_i[None, None] + Dom/2
    a3 = amp_i[None, None]
    p3 = phi_i[None, None]
    
    #compose electric field in the time domain in real form:
    field_om = -a3 * (sin(p3 - oh3*t3) - sin(p3 - ol3*t3)) / t3
    field = np.sum(field_om, -1)
    
    return field


def find_extrema(amp, phi, j_om, Dom):
    '''determines the time-domain extreme value positions of the given electric
    field from its frequency-domain properties by finding the eigenvalues of
    the underlying trigonometric polynomial's companion matrix. [1]
    the frequency grid of the polynomial is required to be equidistant and
    rooted at zero.
    
    [1] J. P. Boyd, J. Eng. Math. 56, 203--219 (2006)
    
    amp  : spectral amplitudes, shape (n_om,)
    phi  : spectral phases in radians, shape (n_om,)
    j_om : indices of zero-centred circular frequency grid, shape (n_om,)
    Dom  : increment of zero-centred circular frequency grid, scalar'''
    
    #prepare indices for Fourier-Frobenius companion matrix:
    n_om_c = j_om[-1]
    j_eye = np.arange(2*n_om_c - 1, dtype=int)
    k_eye = np.arange(1, 2*n_om_c, dtype=int)
    v_eye = np.ones(2*n_om_c-1)
    n_om = len(j_om)
    j_bot = np.ones(2*n_om-1, dtype=int) * (2*n_om_c-1)
    k_bot1 = n_om_c - j_om
    k_bot2 = n_om_c + j_om[:-1]
    k_bot = np.append(k_bot1, k_bot2)
    j_tot = np.append(j_eye, j_bot)
    k_tot = np.append(k_eye, k_bot)
    
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


def chietazeta(para):
    'computes deviation between measured TIPTOE trace and model'
    
    #parse spectral phases:
    phi_hi = para[:n_om_hi]
    phi_lo = para[n_om_hi:]
    
    #find extrema of strong field:
    t_hi = find_extrema(amp_hi, phi_hi, j_om_hi, Dom_hi)
    
    #find extrema of weak field and discard extrema outside delay window:
    t_lo = find_extrema(amp_lo, phi_lo, j_om_lo, Dom_lo)
    b_win_lo = abs(t_lo) < 0.5*Rt
    t_lo = t_lo[b_win_lo]
    
    #evaluate fields' extrema:
    E_hi = efield_re(t_hi[:, None], om_hi, Dom_hi, amp_hi, phi_hi)[:, 0]
    E_lo = efield_re(t_lo[:, None], om_lo, Dom_lo, amp_lo, phi_lo)[:, 0]
    
    #compute fraction of weak pulse within observation window:
    E_los = E_lo**2
    It_cen = 0.5*np.sum((E_los[:-1] + E_los[1:])*np.diff(t_lo))
    print("weak pulse's fluence ratio:", It_cen/It_lo)
    eta = max(q_lo - It_cen/It_lo, 0)/q_lo
    
    #assess whether strong pulse triggers computable amount of ionisation:
    wi_hi = rate_adk(abs(E_hi), IE0)
    if 1 - np.prod(1 - wi_hi) > 0:
        
        #discard positive minima and negative maxima:
        d2E_c = cos(phi_hi[None]) * cos(j_om_hi[None]*Dom_hi*t_hi[:, None])
        d2E_s = sin(phi_hi[None]) * sin(j_om_hi[None]*Dom_hi*t_hi[:, None])
        d2E_hi = -np.sum(amp_hi[None] * (d2E_c + d2E_s) * (j_om_hi[None]*Dom_hi)**2, 1)
        b_keep = E_hi*d2E_hi < 0
        t_hi = t_hi[b_keep]
        E_hi = E_hi[b_keep]
        
        #discard points at low relative field strength:
        b_keep = abs(E_hi) > 0.4*np.max(abs(E_hi))
        t_hi = t_hi[b_keep]

        #set up anchor points for trapezoid composite integration:
        n_trz = 7 #number of trapezoids per shoulder
        inc_trz = Tm_hi/(6*n_trz)
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
        E_hi_trz = efield_re(t_trz[:, None], om_hi, Dom_hi, amp_hi, phi_hi)
        E_lo_trz = efield_re(t_trz[:, None] - delay[None], om_lo, Dom_lo, amp_lo, phi_lo)
        
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
        print("strong pulse's timing:", t_m)
        
        return chi, eta, zeta
        
    else:
        
        #compute fraction of strong pulse within observation window:
        E_his = E_hi**2
        It_cen = 0.5*np.sum((E_his[:-1] + E_his[1:])*np.diff(t_hi))
        mu = 4*(2 - It_cen/It_hi)
        print("strong pulse's fluence ratio:", It_cen/It_hi)
        
        #compute mean time of strong pulse's intensity envelope:
        t_m = np.sum(t_hi * E_his)/np.sum(E_his)
        zeta = abs(t_m)/(0.5*Rt)
        print("strong pulse's timing:", t_m)
        
        return mu, eta, zeta


def fwhm_limits(x, y):
    '''determines the x values that define the full width at half maximum of
    the distribution y. the method is robust with respect to multimodality'''
    
    #find points that intersect the half-maximum of y:
    ymax = np.max(y)
    b_lim = np.diff(np.sign(y - ymax/2))
    
    #identify rising intersection closest to peak:
    j_peak = np.where(y == ymax)[0][0]
    ji_rise = np.where(b_lim > 0)[0]
    dj_rise = ji_rise - j_peak
    j_rise = max(dj_rise[dj_rise < 0]) + j_peak
    
    #identify falling intersection closest to peak:
    ji_fall = np.where(b_lim < 0)[0]
    dj_fall = ji_fall - j_peak
    j_fall = min(dj_fall[dj_fall > 0]) + j_peak
    
    return x[j_rise], x[j_fall]


def parse_snp(file):
    'parses parameters from TIPTOE reconstruction snapshot file'
    
    with open(file, 'r') as snp:
        txt = snp.read()
    start = txt.find('[')
    end = txt.find(']')
    para_s = txt[start+1:end]
    para_s = para_s.replace('\n', ' ')
    para = [float(p) for p in para_s.split(', ')]
    
    return para


#parse optimised parameters from snapshot file:
if check_input:
    para = np.zeros(n_om_hi+n_om_lo)
else:
    para = parse_snp('results/' + file_snp)

#parse spectral phases:
phi_hi = np.unwrap(para[:n_om_hi])
phi_lo = np.unwrap(para[n_om_hi:])

#compute complex electric fields:
dtime = 1
time_plot = np.arange(min(delay) - Rt, max(delay) + Rt + dtime, dtime)
E_rec_hi_c = efield_c(time_plot[:, None], om_hi, Dom_hi, amp_hi, phi_hi)[:, 0]
E_rec_lo_c = efield_c(time_plot[:, None], om_lo, Dom_hi, amp_lo, phi_lo)[:, 0]

#compute absolute electric fields and envelopes:
E_rec_hi = np.real(E_rec_hi_c)
E_rec_lo = np.real(E_rec_lo_c)
E_rec_hi_a = np.abs(E_rec_hi_c)
E_rec_lo_a = np.abs(E_rec_lo_c)

# #find time boundaries that define the intensity envelopes FWHM:
# t_rise_hi, t_fall_hi = fwhm_limits(time_plot, E_rec_hi_a**2)
# fwhm_hi_fs = (t_fall_hi - t_rise_hi)*t_au*1e15
# t_rise_lo, t_fall_lo = fwhm_limits(time_plot, E_rec_lo_a**2)
# fwhm_lo_fs = (t_fall_lo - t_rise_lo)*t_au*1e15

#find extrema of strong field and evaluate local field:
t_hi = find_extrema(amp_hi, phi_hi, j_om_hi, Dom_hi)
E_hi = efield_re(t_hi[:, None], om_hi, Dom_hi, amp_hi, phi_hi)[:, 0]

#discard positive minima and negative maxima:
d2E_c = cos(phi_hi[None]) * cos(j_om_hi[None]*Dom_hi*t_hi[:, None])
d2E_s = sin(phi_hi[None]) * sin(j_om_hi[None]*Dom_hi*t_hi[:, None])
d2E_hi = -np.sum(amp_hi[None] * (d2E_c + d2E_s) * (j_om_hi[None]*Dom_hi)**2, 1)
b_keep = E_hi*d2E_hi < 0
t_hi = t_hi[b_keep]
E_hi = E_hi[b_keep]

#discard points at low relative field strength:
b_keep = abs(E_hi) > 0.4*np.max(abs(E_hi))
t_hi = t_hi[b_keep]

#set up anchor points for trapezoid composite integration:
n_trz = 7 #number of trapezoids per shoulder
inc_trz = Tm_hi/(6*n_trz)
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
E_hi_trz = efield_re(t_trz[:, None], om_hi, Dom_hi, amp_hi, phi_hi)
E_lo_trz = efield_re(t_trz[:, None] - delay[None], om_lo, Dom_lo, amp_lo, phi_lo)

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

# #convert spectra to spectral exposure in J/(m^2 * Hz):
# eps_0_au = e**2 / (a_0*U_au) #atomic unit of vacuum electric permittivity (F/m)
# bins_nu_hi = bins_om_hi/(2*pi*t_au)
# bins_nu_lo = bins_om_lo/(2*pi*t_au)
# nu_hi = (bins_nu_hi[:-1] + bins_nu_hi[1:])/2
# nu_lo = (bins_nu_lo[:-1] + bins_nu_lo[1:])/2
# se_hi_au0 = int_om_hi * eps_0_au * a_0/t_au
# se_lo_au0 = int_om_lo * eps_0_au * a_0/t_au
# se_hi_au = amp_hi**2 * eps_0_au * a_0/t_au
# se_lo_au = amp_lo**2 * eps_0_au * a_0/t_au
# se_hi_si0 = se_hi_au0 * (U_au*t_au)/(a_0**2)
# se_lo_si0 = se_lo_au0 * (U_au*t_au)/(a_0**2)
# se_hi_si = se_hi_au * (U_au*t_au)/(a_0**2)
# se_lo_si = se_lo_au * (U_au*t_au)/(a_0**2)
# Dnu_hi = Dom_hi/(2*pi*t_au)
# Dnu_lo = Dom_lo/(2*pi*t_au)

#plot spectra:
plt.figure(figsize=(6*figfac, 6*figfac))

ax1a = plt.subplot2grid((2, 1), (0, 0))
# ax1a.set_xlim([min(om_spec), max(om_spec)])
ax1a.set_xlim([om_hi[0] - Dom_hi/2, om_hi[-1] + Dom_hi/2])
ax1b = ax1a.twinx()
ax1a.bar(om_hi, amp_hi**2, width=Dom_hi, color='red', edgecolor='white', alpha=0.4)
ax1a.plot(om_spec_hi, spec_om_hi, c='red', lw=1.5*figfac)
# reps = [1] + (n_om_hi-1)*[2] + [1]
# ax1b.plot(np.repeat(bins_om_hi, reps), np.repeat(phi_hi, 2), c='green',
#           lw=0.75*figfac, ls='--')
ax1b.plot(om_hi, phi_hi, color='green', lw=0.5*figfac, drawstyle='steps-mid', ls='--')
ax1b.plot(om_hi, phi_hi, color='green', lw=0, marker='o', markersize=3*figfac,
          markeredgewidth=1.5*figfac, markerfacecolor='none')
ax1a.set_ylim([0, None])
ax1a.set_ylabel('intensity (at. units)', c='red')
ax1b.set_ylabel('phase (radians)', c='green')
ax1a.tick_params(axis='y', colors='red')
ax1b.tick_params(axis='y', colors='green')

ax2a = plt.subplot2grid((2, 1), (1, 0))
ax2a.set_xlim([om_lo[0] - Dom_lo/2, om_lo[-1] + Dom_lo/2])
ax2b = ax2a.twinx()
ax2a.bar(om_lo, amp_lo**2, width=Dom_lo, color='red', edgecolor='white', alpha=0.4)
ax2a.plot(om_spec_lo, spec_om_lo, c='red', lw=1.5*figfac)
# reps = [1] + (n_om_lo-1)*[2] + [1]
# ax2b.plot(np.repeat(bins_om_lo, reps), np.repeat(phi_lo, 2), c='green',
#           lw=0.75*figfac, ls='--')
ax2b.plot(om_lo, phi_lo, color='green', lw=0.5*figfac, drawstyle='steps-mid', ls='--')
ax2b.plot(om_lo, phi_lo, color='green', lw=0, marker='o', markersize=3*figfac,
          markeredgewidth=1.5*figfac, markerfacecolor='none')
ax2a.set_ylim([0, None])
ax2a.set_xlabel('circular frequency (atomic units)')
ax2a.set_ylabel('intensity (at. units)', c='red')
ax2b.set_ylabel('phase (radians)', c='green')
ax2a.tick_params(axis='y', colors='red')
ax2b.tick_params(axis='y', colors='green')
# plt.savefig('{}_nom={}_qlo={}_spectra.png'.format(time_id, n_om, q_lo),
#             bbox_inches='tight', dpi=300)

#plot TIPTOE trace comparison along with reconstructed electric field:
chi, eta, zeta = chietazeta(np.append(phi_hi, phi_lo))
print('total optimisation target:', round(chi + eta + zeta, 4))
plt.figure(figsize=(6*figfac, 6*figfac))
ax1 = plt.subplot2grid((3, 1), (0, 0))
plt.plot(-delay*t_au*1e15, trace, lw=1*figfac, label='input')
plt.plot(-delay*t_au*1e15, ratio_ion, lw=1*figfac, ls='--', label='reconstruction')
# plt.plot(-delay*t_au*1e15, trace, marker='o', lw=0.5*figfac, markersize=2*figfac,
#          markerfacecolor='white', markeredgewidth=1*figfac, ls='--', label='input')
plt.xlim([min(delay_fs), max(delay_fs)])
plt.xlabel('delay (fs)')
plt.ylabel('relative ion yield')
plt.legend(frameon=False, loc=2)
plt.text(0.83, 0.94, r'$\chi = {}$'.format(round(chi, 4)),
          ha='left', va='top', transform=ax1.transAxes)
ax1.xaxis.set_minor_locator(AutoMinorLocator())

ax2 = plt.subplot2grid((3, 1), (1, 0))
# plt.fill_between(time_plot*t_au*1e15, -E_rec_hi_a, E_rec_hi_a, lw=0,
#                  color='red', alpha=0.2)
# plt.plot(time_plot*t_au*1e15, E_rec_hi, lw=4.5*figfac, c='white')
plt.plot(time_plot*t_au*1e15, E_rec_hi, lw=1*figfac, c='red')
plt.plot(time_plot*t_au*1e15, -E_rec_hi_a, lw=0.5*figfac, c='red')
plt.plot(time_plot*t_au*1e15, E_rec_hi_a, lw=0.5*figfac, c='red')
# plt.axvline(t_rise_hi*t_au*1e15, c='black', lw=0.5*figfac, ls='--')
# plt.axvline(t_fall_hi*t_au*1e15, c='black', lw=0.5*figfac, ls='--')
plt.xlim([min(delay_fs), max(delay_fs)])
plt.ylabel(r'$\epsilon_\mathrm{hi}$ (atomic units)')
ax2.xaxis.set_minor_locator(AutoMinorLocator())
# plt.text(0.01, 0.94, r'$\mathrm{FWHM}_I = ' + str(round(fwhm_hi_fs, 2)) + r'$ fs',
#          ha='left', va='top', transform=ax2.transAxes)
plt.text(0.83, 0.94, r'$\zeta = {}$'.format(round(zeta, 4)),
          ha='left', va='top', transform=ax2.transAxes)

ax3 = plt.subplot2grid((3, 1), (2, 0))
# plt.fill_between(time_plot*t_au*1e15, -E_rec_lo_a, E_rec_lo_a, lw=0,
                 # color='dodgerblue', alpha=0.2)
# plt.plot(time_plot*t_au*1e15, E_rec_lo, lw=4.5*figfac, c='white')
plt.plot(time_plot*t_au*1e15, E_rec_lo, lw=1*figfac, c='dodgerblue')
plt.plot(time_plot*t_au*1e15, -E_rec_lo_a, lw=0.5*figfac, c='dodgerblue')
plt.plot(time_plot*t_au*1e15, E_rec_lo_a, lw=0.5*figfac, c='dodgerblue')
# plt.axvline(t_rise_lo*t_au*1e15, c='black', lw=0.5*figfac, ls='--')
# plt.axvline(t_fall_lo*t_au*1e15, c='black', lw=0.5*figfac, ls='--')
plt.xlim([min(delay_fs), max(delay_fs)])
plt.xlabel('time (fs)')
plt.ylabel(r'$\epsilon_\mathrm{lo}$ (atomic units)')
ax3.xaxis.set_minor_locator(AutoMinorLocator())
# plt.text(0.01, 0.94, r'$\mathrm{FWHM}_I = ' + str(round(fwhm_lo_fs, 2)) + r'$ fs',
#          ha='left', va='top', transform=ax3.transAxes)
plt.text(0.83, 0.94, r'$\eta = {}$'.format(round(eta, 4)),
          ha='left', va='top', transform=ax3.transAxes)
plt.tight_layout()
# plt.savefig('{}_nom={}_qlo={}_traces+fields.png'.format(time_id, n_om, q_lo),
#             bbox_inches='tight', dpi=300)

plt.show()
