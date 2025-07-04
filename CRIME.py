#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from numpy import pi, sin, cos, sqrt, exp
from scipy.interpolate import interp1d
from scipy.optimize import differential_evolution
from datetime import datetime

#define time identifier of measurement:
time_id = '250521_1159_4'

#define extraction parameters:
species = 'He'
marg  = 2.5e-3 #margin for cropping periphery of spectra
n_om  = 40 #number of frequencies to consider
q_lo  = 0.95 #minimum fraction of weak pulse's fluence within delay frame
fac_F = 0.8 #fluence scaling factor for strong pulse (focal-volume averaging)

#define parameters of the computation setup:
n_co = 128 #number of processors for multithreading
file = 'tiptoe_h2_{}'.format(time_id)
name = 'tiptoe_h2_{}_{}_nom={}_qlo={}_fac={}'.format(time_id, species, n_om, q_lo, fac_F) #job name

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
d_IE0_eV = {'He':24.587, 'Ne':21.565, 'H2O':12.600}

#set up time increments for intermediate file saves:
dt = 30 #save interval in seconds
start = datetime.now()
now = datetime.now()

#load TIPTOE data:
import h5py
with h5py.File(file + '.h5', 'r') as h5:
    delay_fs = h5['target delays (fs)'][:]
    trace = h5['rel. yield {}+'.format(species)][:]
    lam_spec_hi = h5['wavelengths strong pulse (nm)'][:]
    lam_spec_lo = h5['wavelengths weak pulse (nm)'][:]
    int_lam_hi = h5['spectral intensities strong pulse (arb. u.)'][:]
    int_lam_lo = h5['spectral intensities weak pulse (arb. u.)'][:]
    atts = dict(h5.attrs)
    F_hi = atts['peak fluence strong pulse (J/m^2)']
    F_lo = atts['peak fluence weak pulse (J/m^2)']
    
#apply scaling factor to fluence of strong pulse:
F_hi *= fac_F
    
# #invert delay axis to account for the weak pulse being delayed:
# delay_fs *= -1
    
#centre delay frame:
delay_fs -= np.mean(delay_fs)
    
#convert experimental delay data to atomic units:
delay = delay_fs*1e-15/t_au

#define normalisation factor for deviation between model and experiment:
norm = np.sum((trace - 1)**2)

#convert wavelength to m:
lam_spec_hi *= 1e-9
lam_spec_lo *= 1e-9

#convert spectra from wavelength to circular frequency in SI units:
om_spec_si_hi = 2*pi*c_vac/lam_spec_hi
om_spec_si_lo = 2*pi*c_vac/lam_spec_lo
int_om_hi = int_lam_hi * 2*pi*c_vac/om_spec_si_hi**2
int_om_lo = int_lam_lo * 2*pi*c_vac/om_spec_si_lo**2

#convert to circular frequency in atomic units:
om_spec_hi = om_spec_si_hi*t_au
om_spec_lo = om_spec_si_lo*t_au
int_om_hi /= t_au
int_om_lo /= t_au

#sort by ascending circular frequency:
j_sort_hi = np.argsort(om_spec_hi)
om_spec_hi = om_spec_hi[j_sort_hi]
int_om_hi = int_om_hi[j_sort_hi]
j_sort_lo = np.argsort(om_spec_lo)
om_spec_lo = om_spec_lo[j_sort_lo]
int_om_lo = int_om_lo[j_sort_lo]

#compute time-integrated intensities from peak fluences (at. u.):
Esq_f_hi = 2*F_hi/(c_vac*eps_0) / (t_au*E_au**2)
Esq_f_lo = 2*F_lo/(c_vac*eps_0) / (t_au*E_au**2)

#compute time-integrated intensities from spectral intensities (arb. u.):
Dom_spec_hi = np.gradient(om_spec_hi)
Dom_spec_lo = np.gradient(om_spec_lo)
Esq_i_hi = 2*pi * np.sum(int_om_hi*Dom_spec_hi)
Esq_i_lo = 2*pi * np.sum(int_om_lo*Dom_spec_lo)

#bring spectral intensities to absolute scale:
int_om_hi *= Esq_f_hi/Esq_i_hi
int_om_lo *= Esq_f_lo/Esq_i_lo

#compute normalised cumulative spectral intensities:
cs_hi = np.cumsum(int_om_hi*Dom_spec_hi)
int_hi = cs_hi[-1]
cs_hi /= int_hi
cs_lo = np.cumsum(int_om_lo*Dom_spec_lo)
int_lo = cs_lo[-1]
cs_lo /= int_lo

#identify crop indices for spectra:
j_marg_hi = np.where((cs_hi > marg) & (cs_hi < 1 - marg))[0]
om_min_hi = om_spec_hi[j_marg_hi[0]]
om_max_hi = om_spec_hi[j_marg_hi[-1]]
j_marg_lo = np.where((cs_lo > marg) & (cs_lo < 1 - marg))[0]
om_min_lo = om_spec_lo[j_marg_lo[0]]
om_max_lo = om_spec_lo[j_marg_lo[-1]]

#set up zero-centred circular frequency grid for strong pulse:
Dom_hi = (om_max_hi - om_min_hi)/n_om
j0_bom_hi = round(om_min_hi/Dom_hi) + 0.5
j_bom_hi = np.arange(j0_bom_hi, j0_bom_hi+n_om+1)
j_om_hi = ((j_bom_hi[:-1] + j_bom_hi[1:])/2).astype(int)
Dom_hi = om_min_hi/j0_bom_hi
bins_om_hi = j_bom_hi * Dom_hi
om_hi = (bins_om_hi[:-1] + bins_om_hi[1:])/2

#set up zero-centred circular frequency grid for weak pulse:
Dom_lo = (om_max_lo - om_min_lo)/n_om
j0_bom_lo = round(om_min_lo/Dom_lo) + 0.5
j_bom_lo = np.arange(j0_bom_lo, j0_bom_lo+n_om+1)
j_om_lo = ((j_bom_lo[:-1] + j_bom_lo[1:])/2).astype(int)
Dom_lo = om_min_lo/j0_bom_lo
bins_om_lo = j_bom_lo * Dom_lo
om_lo = (bins_om_lo[:-1] + bins_om_lo[1:])/2

#compute spectral amplitudes on new spectral grids:
i_cs_hi = interp1d(om_spec_hi, cs_hi, bounds_error=False, fill_value=0)
amp_hi = sqrt(np.diff(i_cs_hi(bins_om_hi)) * int_hi/Dom_hi)
i_cs_lo = interp1d(om_spec_lo, cs_lo, bounds_error=False, fill_value=0)
amp_lo = sqrt(np.diff(i_cs_lo(bins_om_lo)) * int_lo/Dom_lo)

#deduce minimum and maximum optical periods of strong field:
T_min = 2*pi/np.max(om_hi)
T_max = 2*pi/np.min(om_hi)

#convert field-free ionisation energy to atomic units:
IE0 = d_IE0_eV[species]*e/U_au

#define range for field-adaptive time grid:
Rt = max(delay) - min(delay)

#prepare time-integrated intensities:
It_hi = 2*pi*np.sum(amp_hi**2)*Dom_hi
It_lo = 2*pi*np.sum(amp_lo**2)*Dom_lo
II_hi = np.sum(amp_hi**2)
II_lo = np.sum(amp_lo**2)


def efield_c(time, bom_i, amp_i, phi_i):
    '''composes the time domain representation of a laser-electric field in its
    complex form based on its properties in the frequency domain. the spectral
    amplitudes and phases are assumed to be constant in between the given
    frequency bins bom_i.
    returns field in complex form
    
    time  : time grid in atomic units, shape (n_t, n_tau)
    bom_i : circular frequency grid in ascending order, shape (n_om+1,)
    amp_i : spectral amplitudes in atomic units, shape (n_om,)
    phi_i : spectral phases in radians, shape (n_om,)'''
    
    #set up three-dimensional variables:
    t3 = time[..., None]
    ol3 = bom_i[None, None, :-1]
    oh3 = bom_i[None, None, 1:]
    a3 = amp_i[None, None]
    p3 = phi_i[None, None]
    
    #compose electric field in the time domain in complex form:
    field_om = a3*1j * (exp(1j*(p3 - oh3*t3)) - exp(1j*(p3 - ol3*t3))) / t3
    field = np.sum(field_om, -1)
    
    return field


def efield_re(time, bom_i, amp_i, phi_i):
    '''composes the time domain representation of a laser-electric field in its
    real form based on its properties in the frequency domain. the spectral
    amplitudes and phases are assumed to be constant in between the given
    frequency bins bom_i.
    returns field in real form
    
    time  : time grid in atomic units, shape (n_t, n_tau)
    bom_i : circular frequency grid in ascending order, shape (n_om+1,)
    amp_i : spectral amplitudes in atomic units, shape (n_om,)
    phi_i : spectral phases in radians, shape (n_om,)'''
    
    #set up three-dimensional variables:
    t3 = time[..., None]
    ol3 = bom_i[None, None, :-1]
    oh3 = bom_i[None, None, 1:]
    a3 = amp_i[None, None]
    p3 = phi_i[None, None]
    
    #compose electric field in the time domain in real form:
    field_om = -a3 * (sin(p3 - oh3*t3) - sin(p3 - ol3*t3)) / t3
    field = np.sum(field_om, -1)
    
    return field
    
    
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


def find_extrema(j_om, amp, phi, dom):
    '''determines the time-domain extreme value positions of the given electric
    field from its frequency-domain properties by finding the eigenvalues of
    the underlying trigonometric polynomial's companion matrix. [1]
    the frequency grid of the polynomial is required to be equidistant and
    rooted at zero.
    
    [1] J. P. Boyd, J. Eng. Math. 56, 203--219 (2006)
    
    j_om : integer circular-frequency factors, shape (n_om,)
    amp  : spectral amplitudes, shape (n_om,)
    phi  : spectral phases in radians, shape (n_om,)
    dom  : circular-frequency increment, scalar'''
    
    #set up Fourier-Frobenius companion matrix:
    n_om = len(j_om)
    n_om_c = j_om[-1]
    a_j = amp * sin(phi) * j_om
    b_j = -amp * cos(phi) * j_om
    A_jk = np.eye(2*n_om_c, k=1, dtype=complex)
    Q = a_j[-1] - 1j*b_j[-1]
    A_jk[-1, :n_om] = -(a_j + 1j*b_j)[::-1]/Q
    A_jk[-1, -n_om+1:] = -(a_j - 1j*b_j)[:-1]/Q
    
    #compute eigenvalues of companion matrix and find extrema in time domain:
    eigen = np.linalg.eigvals(A_jk)
    t_ex = np.angle(eigen)/dom
    e_ex = np.abs(np.abs(eigen) - 1)
    b_crit = e_ex < 1e-3 #discard extrema with large error
    t_ex = np.sort(t_ex[b_crit])
    
    return t_ex


def minfunc(para):
    'computes deviation between measured TIPTOE trace and model'
    
    #parse spectral phases:
    phi_hi = para[:n_om]
    phi_lo = para[n_om:]
    
    #find extrema of strong field and compute absolute local gradient:
    t_hi = find_extrema(j_om_hi, amp_hi, phi_hi, Dom_hi)
    dt_hi = np.abs(np.gradient(t_hi))
    
    #find extrema of weak field and discard extrema outside delay window:
    t_lo = find_extrema(j_om_lo, amp_lo, phi_lo, Dom_lo)
    b_win_lo = abs(t_lo) < 0.5*Rt
    t_lo = t_lo[b_win_lo]
    
    #evaluate fields' extrema:
    E_hi = efield_re(t_hi[:, None], bins_om_hi, amp_hi, phi_hi)[:, 0]
    E_lo = efield_re(t_lo[:, None], bins_om_lo, amp_lo, phi_lo)[:, 0]
    
    #compute fraction of weak pulse within observation window:
    E_los = E_lo**2
    It_cen = 0.5*np.sum((E_los[:-1] + E_los[1:])*np.diff(t_lo))
    eta = max(q_lo - It_cen/It_lo, 0)/q_lo
    
    #assess whether strong pulse triggers computable amount of ionisation:
    wi_hi = rate_adk(abs(E_hi), IE0)
    if 1 - np.prod(1 - wi_hi) > 0:
    
        #discard extrema at low relative field strength:
        E_hi = efield_re(t_hi[:, None], bins_om_hi, amp_hi, phi_hi)[:, 0]
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
        E_hi_trz = efield_re(t_trz[:, None], bins_om_hi, amp_hi, phi_hi)
        E_lo_trz = efield_re(t_trz[:, None] - delay[None], bins_om_lo, amp_lo, phi_lo)
        
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
        
        return chi + eta + zeta
        
    else:
        
        #compute fraction of strong pulse within observation window:
        E_his = E_hi**2
        It_cen = 0.5*np.sum((E_his[:-1] + E_his[1:])*np.diff(t_hi))
        mu = 4*(2 - It_cen/It_hi)
        
        #compute mean time of strong pulse's intensity envelope:
        t_m = np.sum(t_hi * E_his)/np.sum(E_his)
        zeta = abs(t_m)/(0.5*Rt)
        
        return mu + eta + zeta


def callback(xk, convergence):
    global now
    
    delta = (datetime.now() - now).total_seconds()
    if delta > dt:
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
        with open('{}.snp'.format(name), 'w') as txt:
            txt.write(header + para)
        now = datetime.now()


def extract_field(para0=None):
    '''extracts the laser-electric field from a TIPTOE trace employing the ADK
    tunnelling theory to describe the relative ionisation rate. the ionisation
    energy of the target is assumed to be field-invariant. optimises model to
    find spectral amplitude and phase at each point of the given circular-
    frequency grid
    
    para0 : initial guess for parameter vector'''
    
    #set up boundaries for parameter optimisation:
    bounds = 2*n_om*[(0, 2*pi)]
    
    #find global optimum of electric field parameters:
    res = differential_evolution(minfunc, bounds, disp=False, x0=para0,
                                 updating='deferred', maxiter=250000,
                                 workers=max(1, n_co-1), callback=callback,
                                 polish=False, tol=1e-3)
    
    return res


if __name__ == "__main__":
    
    #extract electric field from TIPTOE trace:
    result = extract_field()
    
    #save results:
    header = 'status     : {}'.format(result.message) + '\n'
    header += 'runtime    : {}'.format(datetime.now() - start) + '\n'
    header += 'minfunc(x) : {}'.format(minfunc(result.x)) + '\n'
    par = '[{}'.format(result.x[0])
    for j,x in enumerate(result.x[1:]):
        if not (j+1)%3:
            sep = '\n'
        else:
            sep = ' '
        par += ',{}{}'.format(sep, x)
    par += ']\n'
    with open('{}.snp'.format(name), 'w') as txt:
        txt.write(header + par)
