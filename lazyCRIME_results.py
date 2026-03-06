#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#requirements:
#   * larger delay values correspond to the weak pulse arriving later
#   * strong and weak pulse have identical waveforms (except for a scalar phase offset)

# =============================================================================
# USER INPUT:
# =============================================================================

#define data identifier for HDF5 file with input data:
identifier  = 'inputexample'

#define retrieval parameters:
IE_eV       = 24.587 #(vertical) ionisation energy (eV)
frac        = 0.998 #fraction of total fluence to be covered by frequency grid
q_set       = 0.90 #minimum fraction of weak pulse's fluence within delay/time window
n_om0       = 10 #number of frequency grid points
check_input = False #boolean, sets all spectral phases to 0

# =============================================================================
# LOAD MODULES:
# =============================================================================

import numpy as np
from numpy import pi, sin, cos, sqrt, exp
import h5py
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.patches import Rectangle
figfac = 0.6 #figure scaling factor
plt.rcParams.update({'font.size': 10*figfac})

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

#(vertical) ionisation energies from https://cccbdb.nist.gov:
#He : 24.587 eV
#Ne : 21.565 eV
#Ar : 15.759 eV
#N2 : 15.58 eV

# =============================================================================
# LOAD EXPERIMENTAL DATA:
# =============================================================================

#set up name for snapshot file:
file_snp = 'lazycrime_{}_IE={}eV_nom={}_frac={}_qlo={}.snp'.format(identifier, IE_eV, n_om0, frac, q_set)

#load input data:
file = '{}.h5'.format(identifier)
with h5py.File(file, 'r') as h5:
    print('datasets in {}:'.format(file))
    for key in h5:
        print('  ' + key)
    delay_fs = h5['delay (fs)'][:]
    trace = h5['rel. yield He+'][:]
    wav_spec = h5['wavelength weak pulse (nm)'][:]
    spec = h5['spectral intensity weak pulse (arb. u.)'][:]
    
#centre delay frame:
delay_fs -= np.mean(delay_fs)
    
#convert experimental delay data to atomic units:
delay = delay_fs*1e-15/t_au

#define normalisation factor for deviation between model and experiment:
norm = np.sum((trace - 1)**2)

#define range for field-adaptive time grid:
Rt = max(delay) - min(delay)

#convert field-free ionisation energy to atomic units:
IE = IE_eV*e/U_au

# =============================================================================
# FREQUENCY GRID:
# =============================================================================


def frac_indices(tr, frac):
    '''identifies the indices in the given trace that amount to the specified
    fractional cumulative sum
    
    tr   : one-dimensional distribution
    frac : fraction of the cumulative sum to be included [0; 1]'''
    
    #sort indices following decending trace values and compute cumulative sum:
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
wav_spec = wav_spec[j_sort]
spec = spec[j_sort]

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
Dom_spec = np.gradient(om_spec)
amp = sqrt(spec_om_maj/(Dom*np.sum(spec_om_maj)))
spec_om /= np.sum(Dom_spec*spec_om)

#compute mean frequency and optical period:
om0 = np.sum(om*amp**2)/np.sum(amp**2)
Tm = 2*pi*np.sum(amp**2 / om)/np.sum(amp**2)

# =============================================================================
# LASER-ELECTRIC FIELD AND TUNNELLING RATE:
# =============================================================================


def efield_c(time, om_i, Dom, amp_i, phi_i):
    '''composes the time domain representation of a laser-electric field in its
    complex form based on its properties in the frequency domain. the spectral
    amplitudes and phases are assumed to be constant within the frequency
    increments defined by om_i and Dom.
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
    amplitudes and phases are assumed to be constant within the frequency
    increments defined by om_i and Dom.
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
    field from its frequency-domain properties, by finding the eigenvalues of
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
    
    E_abs : absolute electric field strength in atomic units, shape (n_t,)
    IE    : ionisation energy in atomic units, scalar'''
            
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


def chiqtm(para):
    'quantifies deviation between measured TIPTOE trace and model'
    
    #parse spectral phases and delay/phase offset:
    t0_lo, off_lo, F_hi, ratio_F = para[:4]
    phi_hi = np.array(para[4:])
    phi_lo = phi_hi + (om - om0)*t0_lo + off_lo
    
    #derive absolute amplitudes of strong and weak pulse:
    amp_hi = amp * sqrt(F_hi)
    amp_lo = amp_hi/sqrt(ratio_F)
    
    #find extrema of strong field:
    t_hi = find_extrema(amp_hi, phi_hi, j_om, Dom)
    
    #find extrema of weak field and discard extrema outside delay window:
    t_lo = find_extrema(amp_lo, phi_lo, j_om, Dom)
    b_win_lo = abs(t_lo) < 0.5*Rt
    t_lo = t_lo[b_win_lo]
    
    #evaluate fields at extrema:
    E_hi = efield_re(t_hi[:, None], om, Dom, amp_hi, phi_hi)[:, 0]
    E_lo = efield_re(t_lo[:, None], om, Dom, amp_lo, phi_lo)[:, 0]
    
    #compute fraction of weak pulse within observation window:
    E_los = E_lo**2
    It_cen = 0.5*np.sum((E_los[:-1] + E_los[1:])*np.diff(t_lo))
    
    #assess whether strong pulse triggers computable amount of ionisation:
    wi_hi = rate_adk(abs(E_hi), IE)
    if 1 - np.prod(1 - wi_hi) > 0:
        
        #discard positive minima and negative maxima:
        d2E_c = cos(phi_hi[None]) * cos(j_om[None]*Dom*t_hi[:, None])
        d2E_s = sin(phi_hi[None]) * sin(j_om[None]*Dom*t_hi[:, None])
        d2E_hi = -np.sum(amp_hi[None] * (d2E_c + d2E_s) * (j_om[None]*Dom)**2, 1)
        b_keep = E_hi*d2E_hi < 0
        t_hi = t_hi[b_keep]
        E_hi = E_hi[b_keep]
        
        #discard points at low relative field strength:
        b_keep = abs(E_hi) > 0.4*np.max(abs(E_hi))
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
        E_hi_trz = efield_re(t_trz[:, None], om, Dom, amp_hi, phi_hi)
        E_lo_trz = efield_re(t_trz[:, None] - delay[None], om, Dom, amp_lo, phi_lo)
        
        #compute instantaneous ionisation rates:
        wi_ref = rate_adk(abs(E_hi_trz[:, 0]), IE)
        wi_sum = rate_adk(abs(E_hi_trz + E_lo_trz), IE)
        
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
        
        return chi, It_cen/(2*pi*F_hi/ratio_F), t_m
        
    else:
        
        #compute fraction of strong pulse within observation window:
        E_his = E_hi**2
        It_cen = 0.5*np.sum((E_his[:-1] + E_his[1:])*np.diff(t_hi))
        mu = 4*(2 - It_cen/(2*pi*F_hi))
        
        #compute mean time of strong pulse's intensity envelope:
        t_m = np.sum(t_hi * E_his)/np.sum(E_his)
        
        return mu, It_cen/(2*pi*F_hi/ratio_F), t_m


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
    para = [0, 0, 1, 1000] + n_om*[0]
else:
    para = parse_snp(file_snp)

#parse spectral phases and delay/phase offset:
t0_lo, off_lo, F_hi, ratio_F = para[:4]
phi_hi = np.array(para[4:])
phi_lo = phi_hi + (om - om0)*t0_lo + off_lo

#derive absolute amplitudes of strong and weak pulse:
amp_hi = amp * sqrt(F_hi)
amp_lo = amp_hi/sqrt(ratio_F)

#unwrap spectral phases:
phi_hi = np.unwrap(phi_hi)
phi_lo = np.unwrap(phi_lo)

#compute complex electric fields:
dtime = 1
time_plot = np.arange(min(delay) - Rt, max(delay) + Rt + dtime, dtime)
E_rec_hi_c = efield_c(time_plot[:, None], om, Dom, amp_hi, phi_hi)[:, 0]
E_rec_lo_c = efield_c(time_plot[:, None], om, Dom, amp_lo, phi_lo)[:, 0]

#compute absolute electric fields and envelopes:
E_rec_hi = np.real(E_rec_hi_c)
E_rec_lo = np.real(E_rec_lo_c)
E_rec_hi_a = np.abs(E_rec_hi_c)
E_rec_lo_a = np.abs(E_rec_lo_c)

#find time boundaries that define the intensity envelopes FWHM:
t_rise_hi, t_fall_hi = fwhm_limits(time_plot, E_rec_hi_a**2)
fwhm_hi_fs = (t_fall_hi - t_rise_hi)*t_au*1e15
t_rise_lo, t_fall_lo = fwhm_limits(time_plot, E_rec_lo_a**2)
fwhm_lo_fs = (t_fall_lo - t_rise_lo)*t_au*1e15

#find extrema of strong field and evaluate local field:
t_hi = find_extrema(amp_hi, phi_hi, j_om, Dom)
E_hi = efield_re(t_hi[:, None], om, Dom, amp_hi, phi_hi)[:, 0]

#discard positive minima and negative maxima:
d2E_c = cos(phi_hi[None]) * cos(j_om[None]*Dom*t_hi[:, None])
d2E_s = sin(phi_hi[None]) * sin(j_om[None]*Dom*t_hi[:, None])
d2E_hi = -np.sum(amp_hi[None] * (d2E_c + d2E_s) * (j_om[None]*Dom)**2, 1)
b_keep = E_hi*d2E_hi < 0
t_hi = t_hi[b_keep]
E_hi = E_hi[b_keep]

#discard points at low relative field strength:
b_keep = abs(E_hi) > 0.4*np.max(abs(E_hi))
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
E_hi_trz = efield_re(t_trz[:, None], om, Dom, amp_hi, phi_hi)
E_lo_trz = efield_re(t_trz[:, None] - delay[None], om, Dom, amp_lo, phi_lo)

#compute instantaneous ionisation rates:
wi_ref = rate_adk(abs(E_hi_trz[:, 0]), IE)
wi_sum = rate_adk(abs(E_hi_trz + E_lo_trz), IE)

#integrate along realtime axis to obtain cation populations:
wm_ref = 0.5*(wi_ref[:-1] + wi_ref[1:])
Pn_ref = exp(-np.sum(wm_ref*np.diff(t_trz)))
wm_sum = 0.5*(wi_sum[:-1] + wi_sum[1:])
Pn_sum = exp(-np.sum(wm_sum*np.diff(t_trz)[:, None], 0))
Pc_ref = 1 - Pn_ref
Pc_sum = 1 - Pn_sum

#compute ionisation ratio and deviation between model and experiment:
ratio_ion = Pc_sum/Pc_ref

#bring spectral intensities to absolute scale:
F_lo = F_hi/ratio_F
spec_om_hi = spec_om*F_hi
spec_om_lo = spec_om*F_lo

#compute spectral exposure in J/(m^2 * nm):
nu_l = (om - Dom/2)/(2*pi*t_au)
nu_g = (om + Dom/2)/(2*pi*t_au)
nu_spec_l = (om_spec - Dom_spec/2)/(2*pi*t_au)
nu_spec_g = (om_spec + Dom_spec/2)/(2*pi*t_au)
wav_l = c_vac*1e9/nu_g
wav_g = c_vac*1e9/nu_l
Dwav = wav_g - wav_l
wav_spec_l = c_vac*1e9/nu_spec_g
wav_spec_g = c_vac*1e9/nu_spec_l
Dwav_spec = wav_spec_g - wav_spec_l
pre = pi * c_vac * eps_0 * t_au * E_au**2
se_spec_hi = pre * spec_om_hi * Dom_spec/Dwav_spec
se_spec_lo = pre * spec_om_lo * Dom_spec/Dwav_spec
se_hi = pre * amp_hi**2 * Dom/Dwav
se_lo = pre * amp_lo**2 * Dom/Dwav

#split frequency bands into islands:
j_break = np.where(np.diff(om) > 1.5*Dom)[0] + 1
j_split = np.split(np.arange(len(om)), j_break)

#set up wavelength limits:
j_lamlo = np.argmin(wav_spec)
j_lamhi = np.argmax(wav_spec)
dwav_spec = abs(np.gradient(wav_spec))
lim_lam = [wav_spec[j_lamlo] - dwav_spec[j_lamlo]/2,
           wav_spec[j_lamhi] + dwav_spec[j_lamhi]/2]

#plot spectra:
fig = plt.figure(figsize=(5.5*figfac, 4*figfac))
fig.subplots_adjust(hspace=0.1)

ax1a = plt.subplot2grid((2, 1), (0, 0))
ax1a.set_xlim(lim_lam)
ax1b = ax1a.twinx()
for j in range(n_om):
    wl_j = 2*pi*c_vac*1e9*t_au/(om[j] + Dom/2)
    wr_j = 2*pi*c_vac*1e9*t_au/(om[j] - Dom/2)
    rec_j = Rectangle((wl_j, 0), wr_j - wl_j, se_hi[j], lw=0.5*figfac, ec='black', fc='red', alpha=0.4)
    ax1a.add_artist(rec_j)
ax1a.plot(wav_spec, se_spec_hi, c='red', lw=1.5*figfac)
for jj in j_split:
    wl_jj = 2*pi*c_vac*1e9*t_au/(om[jj] - Dom/2)
    wr_jj = 2*pi*c_vac*1e9*t_au/(om[jj] + Dom/2)
    w_jj = np.ravel(np.stack((wl_jj[:, None], wr_jj[:, None]), 1))
    phi_jj = np.repeat(phi_hi[jj], 2)
    ax1b.plot(w_jj, phi_jj, c='green', lw=1*figfac)
ax1a.set_ylim([0, np.max(se_spec_hi)*1.05])
ax1a.set_ylabel(r'$H_{\mathrm{e}, \lambda, \mathrm{hi}}$ $\left(\frac{\mathrm{J}}{\mathrm{m}^2\mathrm{nm}}\right)$', c='red')
ax1b.set_ylabel(r'$\phi_\mathrm{hi}$ (radians)', c='green')
ax1a.xaxis.set_minor_locator(AutoMinorLocator())
ax1a.yaxis.set_minor_locator(AutoMinorLocator())
ax1b.yaxis.set_minor_locator(AutoMinorLocator())
ax1a.tick_params(axis='y', colors='red', which='both')
ax1b.tick_params(axis='y', colors='green', which='both')
ax1a.set_xticklabels([])

ax2a = plt.subplot2grid((2, 1), (1, 0))
ax2a.set_xlim(lim_lam)
ax2b = ax2a.twinx()
for j in range(n_om):
    wl_j = 2*pi*c_vac*1e9*t_au/(om[j] + Dom/2)
    wr_j = 2*pi*c_vac*1e9*t_au/(om[j] - Dom/2)
    rec_j = Rectangle((wl_j, 0), wr_j - wl_j, se_lo[j], lw=0.5*figfac, ec='black', fc='red', alpha=0.4)
    ax2a.add_artist(rec_j)
ax2a.plot(wav_spec, se_spec_lo, c='red', lw=1.5*figfac)
for jj in j_split:
    wl_jj = 2*pi*c_vac*1e9*t_au/(om[jj] - Dom/2)
    wr_jj = 2*pi*c_vac*1e9*t_au/(om[jj] + Dom/2)
    w_jj = np.ravel(np.stack((wl_jj[:, None], wr_jj[:, None]), 1))
    phi_jj = np.repeat(phi_lo[jj], 2)
    ax2b.plot(w_jj, phi_jj, c='green', lw=1*figfac)
ax2a.set_ylim([0, np.max(se_spec_lo)*1.05])
ax2a.set_xlabel(r'$\lambda$ (nm)')
ax2a.set_ylabel(r'$H_{\mathrm{e}, \lambda, \mathrm{lo}}$ $\left(\frac{\mathrm{J}}{\mathrm{m}^2\mathrm{nm}}\right)$', c='red')
ax2b.set_ylabel(r'$\phi_\mathrm{lo}$ (radians)', c='green')
ax2a.xaxis.set_minor_locator(AutoMinorLocator())
ax2a.yaxis.set_minor_locator(AutoMinorLocator())
ax2b.yaxis.set_minor_locator(AutoMinorLocator())
ax2a.tick_params(axis='y', colors='red', which='both')
ax2b.tick_params(axis='y', colors='green', which='both')

#plot TIPTOE trace comparison along with reconstructed electric field:
chi, q_lo_act, t_m = chiqtm(para)
plt.figure(figsize=(6*figfac, 6*figfac))
ax1 = plt.subplot2grid((3, 1), (0, 0))
plt.plot(delay*t_au*1e15, trace, lw=1*figfac, c='mediumvioletred', alpha=0.75, label='input')
plt.plot(delay*t_au*1e15, ratio_ion, lw=1*figfac, c='darkorange', alpha=0.75, label='reconstruction')
plt.xlim([min(delay_fs), max(delay_fs)])
plt.xlabel(r'$\tau$ (fs)')
plt.ylabel(r'$Q$')
plt.legend(frameon=False, loc=2)
plt.text(0.98, 0.94, r'$\chi = {}$'.format(round(chi, 4)),
          ha='right', va='top', transform=ax1.transAxes)
ax1.xaxis.set_minor_locator(AutoMinorLocator())

ax2 = plt.subplot2grid((3, 1), (1, 0))
plt.plot(-time_plot*t_au*1e15, E_rec_hi, lw=1*figfac, c='red')
plt.plot(-time_plot*t_au*1e15, -E_rec_hi_a, lw=0.5*figfac, c='red')
plt.plot(-time_plot*t_au*1e15, E_rec_hi_a, lw=0.5*figfac, c='red')
plt.axvline(-t_rise_hi*t_au*1e15, c='black', lw=0.5*figfac, ls='--')
plt.axvline(-t_fall_hi*t_au*1e15, c='black', lw=0.5*figfac, ls='--')
plt.xlim([min(delay_fs), max(delay_fs)])
plt.ylabel(r'$\epsilon_\mathrm{hi}$ (at. units)')
ax2.xaxis.set_minor_locator(AutoMinorLocator())
plt.text(0.01, 0.94, r'$\mathrm{FWHM}_I = ' + str(round(fwhm_hi_fs, 1)) + r'$ fs',
         ha='left', va='top', transform=ax2.transAxes)
plt.text(0.98, 0.94, r'$\left<t\right>_\mathrm{hi} = $' + '${}$ fs'.format(round(t_m*t_au*1e15, 2)),
          ha='right', va='top', transform=ax2.transAxes)

ax3 = plt.subplot2grid((3, 1), (2, 0))
plt.plot(-time_plot*t_au*1e15, E_rec_lo, lw=1*figfac, c='dodgerblue')
plt.plot(-time_plot*t_au*1e15, -E_rec_lo_a, lw=0.5*figfac, c='dodgerblue')
plt.plot(-time_plot*t_au*1e15, E_rec_lo_a, lw=0.5*figfac, c='dodgerblue')
plt.axvline(-t_rise_lo*t_au*1e15, c='black', lw=0.5*figfac, ls='--')
plt.axvline(-t_fall_lo*t_au*1e15, c='black', lw=0.5*figfac, ls='--')
plt.xlim([min(delay_fs), max(delay_fs)])
plt.xlabel(r'$-t$ (fs)')
plt.ylabel(r'$\epsilon_\mathrm{lo}$ (at. units)')
ax3.xaxis.set_minor_locator(AutoMinorLocator())
plt.text(0.01, 0.94, r'$\mathrm{FWHM}_I = ' + str(round(fwhm_lo_fs, 1)) + r'$ fs',
         ha='left', va='top', transform=ax3.transAxes)
plt.text(0.98, 0.94, r'$q = $' + '${}$'.format(round(q_lo_act, 2)),
          ha='right', va='top', transform=ax3.transAxes)
plt.tight_layout()
plt.show()
