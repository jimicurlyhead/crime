#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from numpy import pi, sin, cos, sqrt, log, exp
import h5py
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
figfac = 0.75
plt.rcParams.update({'font.size': 8*figfac})


#requirements:
#   * shapes of strong and weak fields' waveforms are identical
#   * larger delay values correspond to the weak pulse arriving later

#define retrieval parameters:
file         = 'tiptoe_wl_wwATAS' #name of HDF5 file with raw data
n_om         = 40 #number of frequency bands used for waveform retrieval
frac         = 0.99 #fraction of total fluence to be covered by frequency grid
IE0_eV       = 9 #ionisation energy (eV)

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
# LOAD EXPERIMENTAL DATA AND OPTIMISED PARAMETERS:
# =============================================================================


def parse_snp(file):
    'parses parameters from lazyCRIME retrieval snapshot file'
    
    with open(file, 'r') as snp:
        txt = snp.read()
        
    #parse optimised parameters:
    start = txt.find('[')
    end = txt.find(']')
    para_s = txt[start+1:end]
    para_s = para_s.replace('\n', ' ')
    para = [float(p) for p in para_s.split(', ')]
    
    return para


#set up name for snapshot file:
file_snp = file + '_nom={}_frac={}_IE={}eV'.format(n_om, frac, IE0_eV)

#parse optimised parameters from snapshot file:
para = parse_snp(file_snp + '.snp')
delay0, F_hi, ratio_F = para[:3]
phi = np.array(para[3:])

#load TIPTOE data:
with h5py.File(file + '.h5', 'r') as h5:
    # print('datasets in {}:'.format(file + '.h5'))
    # for key in h5:
    #     print('  ' + key)
    delay_fs = h5['delays (fs)'][:]
    trace = h5['rel. white-light yield'][:]
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
dom_spec = abs(np.gradient(om_spec))
spec_om = spec * 2*pi*c_vac*t_au/om_spec**2

#identify wavelength regions that amount to majority of fluence:
j_maj = frac_indices(spec_om*dom_spec, frac)

#set up zero-centred circular frequency grid that covers whole spectrum:
om_a = min((om_spec - dom_spec/2)[j_maj])
om_b = max((om_spec + dom_spec/2)[j_maj])
Dom = np.sum(dom_spec[j_maj])/n_om
j_om = np.arange(int(np.ceil(om_b/Dom)))
om = j_om * Dom
j_bom = np.arange(int(np.ceil(om_b/Dom))+1) - 0.5
bins_om = j_bom * Dom

#project input spectrum onto compressed circular frequency grid:
weights = spec_om * dom_spec/Dom
spec_om_maj = np.histogram(om_spec, weights=weights, bins=bins_om)[0]

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


def efield_c(time, om_i, dom, amp_i, phi_i):
    '''composes the time domain representation of a laser-electric field in its
    complex form based on its properties in the frequency domain. the spectral
    amplitudes and phases are assumed to be constant across the frequency bins.
    returns field in complex form
    
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
    
    #compose electric field in the time domain in complex form:
    field_om = a3*1j * (exp(1j*(p3 - oh3*t3)) - exp(1j*(p3 - ol3*t3))) / t3
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
# COMPUTE RECONSTRUCTED WAVEFORMS:
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
    
    #find extrema of strong field and compute absolute local gradient:
    t_hi = find_extrema(amp_hi, phi)
    dt_hi = np.abs(np.gradient(t_hi))
    
    #evaluate field extrema:
    E_hi = efield_re(t_hi[:, None], om, Dom, amp_hi, phi)[:, 0]
    
    #assess whether strong pulse triggers computable amount of ionisation:
    wi_hi = rate_adk(abs(E_hi), IE0)
    if 1 - np.prod(1 - wi_hi) > 0:
    
        #discard extrema at low relative field strength:
        b_keep = abs(E_hi) > 0.5*np.max(abs(E_hi))
        t_hi = t_hi[b_keep]
        dt_hi = dt_hi[b_keep]
        
        #set up anchor points for trapezoid composite integration:
        Rt_trz = dt_hi/3 #local-wavelength scaled grid radius
        n_trz = 7 #number of trapezoids per shoulder
        dt_anc = np.arange(-n_trz, n_trz+1)[None] * Rt_trz[:, None]/n_trz
        t_trz2 = t_hi[:, None] + dt_anc
        fac0_trz = np.ones(2*n_trz+1)
        fac0_trz[[0, -1]] = 0.5
        fac_trz2 = fac0_trz[None] * Rt_trz[:, None]/n_trz
        t_trz = np.ravel(t_trz2)
        fac_trz = np.ravel(fac_trz2)
    
        #evaluate electric fields on field-adapted grid:
        E_hi_trz = efield_re(t_trz[:, None], om, Dom, amp_hi, phi)
        E_lo_trz = efield_re(t_trz[:, None] - delay_o[None], om, Dom, amp_lo, phi)
        
        #compute instantaneous ionisation rates:
        wi_ref = rate_adk(abs(E_hi_trz[:, 0]), IE0)
        wi_sum = rate_adk(abs(E_hi_trz + E_lo_trz), IE0)
        
        #integrate along realtime axis to obtain cation populations:
        Pn_ref = np.prod(1 - wi_ref*fac_trz)
        Pn_sum = np.prod(1 - wi_sum*fac_trz[:, None], 0)
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


#derive absolute amplitudes of strong and weak pulse:
amp_hi = amp * sqrt(F_hi)
amp_lo = amp_hi/sqrt(ratio_F)

#compute offset delay grid:
delay_o = delay - delay0

#compute electric field:
dtime = 1
time_plot = np.arange(min(delay), max(delay) + dtime, dtime) + delay0
E_rec_c = efield_c(time_plot[:, None], om, Dom, amp_hi, phi)[:, 0]
E_rec = np.real(E_rec_c)
E_rec_a = np.abs(E_rec)

#find time boundaries that define the intensity envelope's FWHM:
try:
    t_rise, t_fall = fwhm_limits(time_plot, E_rec_a**2)
    fwhm_fs = (t_fall - t_rise)*t_au*1e15
except:
    t_rise, t_fall = time_plot[[0, -1]]
    fwhm_fs = np.nan

#find extrema of strong field and compute absolute local gradient:
t_hi = find_extrema(amp_hi, phi)
dt_hi = np.abs(np.gradient(t_hi))

#evaluate field extrema:
E_hi = efield_re(t_hi[:, None], om, Dom, amp_hi, phi)[:, 0]

#discard extrema at low relative field strength:
b_keep = abs(E_hi) > 0.5*np.max(abs(E_hi))
t_hi = t_hi[b_keep]
dt_hi = dt_hi[b_keep]

#set up anchor points for trapezoid composite integration:
Rt_trz = dt_hi/3 #local-wavelength scaled grid radius
n_trz = 7 #number of trapezoids per shoulder
dt_anc = np.arange(-n_trz, n_trz+1)[None] * Rt_trz[:, None]/n_trz
t_trz2 = t_hi[:, None] + dt_anc
fac0_trz = np.ones(2*n_trz+1)
fac0_trz[[0, -1]] = 0.5
fac_trz2 = fac0_trz[None] * Rt_trz[:, None]/n_trz
t_trz = np.ravel(t_trz2)
fac_trz = np.ravel(fac_trz2)

#evaluate electric fields on field-adapted grid:
E_hi_trz = efield_re(t_trz[:, None], om, Dom, amp_hi, phi)
E_lo_trz = efield_re(t_trz[:, None] - delay_o[None], om, Dom, amp_lo, phi)

#compute instantaneous ionisation rates:
wi_ref = rate_adk(abs(E_hi_trz[:, 0]), IE0)
wi_sum = rate_adk(abs(E_hi_trz + E_lo_trz), IE0)

#integrate along realtime axis to obtain cation populations:
Pn_ref = np.prod(1 - wi_ref*fac_trz)
Pn_sum = np.prod(1 - wi_sum*fac_trz[:, None], 0)
Pc_ref = 1 - Pn_ref
Pc_sum = 1 - Pn_sum

#compute ionisation ratio and deviation between model and experiment:
ratio_ion = Pc_sum/Pc_ref            
chi = np.sum((trace - ratio_ion)**2)/norm

#compute mean time of ionisation:
t_m = np.sum(t_trz * wi_ref)/np.sum(wi_ref)
zeta = abs(t_m)/(0.5*Rt)

#plot input spectrum, frequency-grid representation, and spectral phases:
dwav_spec = abs(np.gradient(wav_spec))
plt.figure(figsize=(5*figfac, 3.5*figfac))

plt.subplot2grid((3, 1), (0, 0))
plt.fill_between(wav_spec, 0, spec, step='mid', lw=0, fc='red', alpha=0.5)
plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
plt.xlim(wav_spec[0] - dwav_spec[0]/2, wav_spec[-1] + dwav_spec[-1]/2)
plt.ylim(0, max(spec)*1.05)
plt.gca().set_xticklabels([])
plt.ylabel(r'$I_{\lambda, \mathrm{input}}$')

plt.subplot2grid((3, 1), (1, 0))
for j in range(len(wav)):
    plt.plot(2*[wav[j]], [0, spec_wav_maj[j]], c='red', lw=0.5*figfac, alpha=0.5)
    wl = 2*pi*c_vac*1e9*t_au/(om[j] + Dom/2)
    wr = 2*pi*c_vac*1e9*t_au/(om[j] - Dom/2)
    plt.plot([wl, wr], 2*[spec_wav_maj[j]], c='red', lw=2.5*figfac, solid_capstyle='butt')
plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
plt.xlim(wav_spec[0] - dwav_spec[0]/2, wav_spec[-1] + dwav_spec[-1]/2)
plt.ylim([0, max(spec)*1.05])
plt.gca().set_xticklabels([])
plt.ylabel(r'$I_{\lambda, \mathrm{grid}}$')

plt.subplot2grid((3, 1), (2, 0))
for j in range(len(wav)):
    plt.plot(2*[wav[j]], [0, phi[j]], c='green', lw=0.5*figfac, alpha=0.5)
    wl = 2*pi*c_vac*1e9*t_au/(om[j] + Dom/2)
    wr = 2*pi*c_vac*1e9*t_au/(om[j] - Dom/2)
    plt.plot([wl, wr], 2*[phi[j]], c='green', lw=2.5*figfac, solid_capstyle='butt')
plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
plt.xlim(wav_spec[0] - dwav_spec[0]/2, wav_spec[-1] + dwav_spec[-1]/2)
plt.ylim([0, None])
plt.xlabel(r'$\lambda$ (nm)')
plt.ylabel(r'$\phi$')
plt.show()

#plot TIPTOE trace comparison along with reconstructed electric field:
lim_t = [min(delay_fs), max(delay_fs)]
fig = plt.figure(figsize=(5*figfac, 3*figfac))
fig.subplots_adjust(hspace=0.8*figfac)
ax1 = plt.subplot2grid((2, 1), (0, 0))
plt.plot(delay*t_au*1e15, trace, lw=1*figfac, c='black', ls='-')
plt.plot(delay*t_au*1e15, ratio_ion, lw=1*figfac, c='red', ls='--')
plt.xlim(lim_t)
plt.xlabel(r'$\tau$ (fs)')
plt.ylabel(r'$Q$')
plt.text(0.02, 0.95, r'$\chi = {}$'.format(round(chi, 3)),
          ha='left', va='top', transform=ax1.transAxes)
plt.text(0.02, 0.02, r'$\zeta = {}$'.format(round(zeta, 3)),
          ha='left', va='bottom', transform=ax1.transAxes)
ax1.xaxis.set_minor_locator(AutoMinorLocator())

ax2 = plt.subplot2grid((2, 1), (1, 0))
plt.fill_between(-(time_plot - delay0)*t_au*1e15, -E_rec_a, E_rec_a, lw=0,
                 color='red', alpha=0.1)
plt.plot(-(time_plot - delay0)*t_au*1e15, E_rec, lw=1*figfac, c='red')
plt.plot(-(time_plot - delay0)*t_au*1e15, -E_rec_a, lw=0.5*figfac, c='red', ls='--')
plt.plot(-(time_plot - delay0)*t_au*1e15, E_rec_a, lw=0.5*figfac, c='red', ls='--')
plt.axvline(-(t_rise - delay0)*t_au*1e15, c='black', lw=0.5*figfac, ls='--')
plt.axvline(-(t_fall - delay0)*t_au*1e15, c='black', lw=0.5*figfac, ls='--')
plt.xlim(lim_t)
plt.ylim(-max(abs(E_rec))*1.2, max(abs(E_rec))*1.2)
plt.xlabel(r'$-t$ (fs)')
plt.ylabel(r'$\epsilon_\mathrm{hi}$ (at. u.)')
ax2.xaxis.set_minor_locator(AutoMinorLocator())
plt.text(0.02, 0.95, r'$\mathrm{FWHM}_I = ' + str(round(fwhm_fs, 2)) + r'$ fs',
         ha='left', va='top', transform=ax2.transAxes)
plt.show()
