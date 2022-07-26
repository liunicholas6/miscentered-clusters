import numpy as np
from scipy.integrate import quad
from colossus.cosmology import cosmology
from colossus.halo import profile_einasto, splashback, mass_so, concentration, profile_diemer22
from colossus.lss import peaks
cosmology.setCosmology('planck15')

# FUNCTION FOR EINASTO PROFILE GIVEN PARAMETERS
def Einasto(r, rho_s, r_s, alpha):
    exp_arg = -2/alpha*(((r/r_s)**alpha)-1)
    rho_EIN = rho_s*np.exp(exp_arg)
    return rho_EIN

# FUNCTION TO INTEGRATE PROFILE TO GET MSP GIVEN EINASTO PROFILE:
def Msp_int(Rsp,rho_s, r_s, alpha):
    def Integrand(r,rho_s, r_s, alpha):
        return Einasto(r, rho_s, r_s, alpha)*4*np.pi*(r**2)
    Msp = quad(Integrand, 0, Rsp, args = (rho_s, r_s, alpha))
    return Msp[0]

# FUNCTION TO GET PEAK HEIGHT FROM ALPHA USING GAO ET AL RELATION:
def nu_func(alpha):
    # alpha = 0.155 + .0095v^2 # THIS IS THE RELATION
    v = np.sqrt((alpha - 0.155)/.0095)
    return v

# FUNCTION TO GET R200M FROM SPLASHBACK RADIUS AND PEAK HEIGHT USING MORE ET AL 2015: 
def R200m_func(Rsp, nu, z):
    Rsp_over_R200m = splashback.modelMore15RspR200m(nu200m=nu, z=z, Gamma=None, statistic='mean')
    R200m = Rsp/Rsp_over_R200m
    return R200m # in kpc/h if Rsp is in kpc/h

# FUNCTION FOR METHOD 1 OF GETTING SPLASHBACK MASS:
def Method1(Rsp, alpha, z): #Rsp in Mpc/h?
    # 1
    nu = nu_func(alpha)
    #print('nu = ',nu)
    # 2
    R200m = R200m_func(Rsp, nu, z) 
    # 3
    M200m = mass_so.R_to_M(R200m, z, mdef = '200m')
    # 4
    #c = 10**(-.125*np.log10(m) + 2.372) #add in h term? # Gao et al 2008 for c200 # could use colossus but its 200c?
    c,mask = concentration.modelBhattacharya13(M200m, z, mdef = '200m')
    # 5
    p_einasto = profile_einasto.EinastoProfile(M = M200m, c = c, z = z, mdef = '200m')
    rhos = p_einasto.par['rhos'] # in Msun*h^2/kpc^3
    rs = p_einasto.par['rs'] # in kpc
    alpha_new = p_einasto.par['alpha']
    # 6
    Msp = Msp_int(Rsp, rhos, rs, alpha)
    return Msp #np.format_float_scientific(Msp)
    
# FUNCTION FOR METHOD 2 OF GETTING SPLASHBACK MASS:
def Method2(Rsp, alpha, z): #Rsp in Mpc/h?
    # 1
    nu = nu_func(alpha)
    # 2
    R200m = R200m_func(Rsp, nu, z) 
    # 3
    M200m = mass_so.R_to_M(R200m, z, mdef = '200m')
    # 4
    Msp_over_M200m = splashback.modelMore15MspM200m(nu200m=nu, z=z, Gamma=None, statistic='mean')
    #Msp_over_M200m = splashback.modelDiemer20MspM200m(nu200m=nu, z=z, rspdef='mean')
    
    Msp = Msp_over_M200m*M200m
    
    return Msp