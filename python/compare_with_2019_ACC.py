import numpy as np

def get_a_1():
    alpha_0=10
    k_G=0.1
    z_G=10
    p_G=0.01

    a_1 = lambda s: (
        (alpha_0+1/k_G)*s + (alpha_0*z_G+p_G/k_G))/(
        (1+1/k_G)*s+(z_G+p_G/k_G))
    return a_1


a_1 = get_a_1()

def get_a_2():
    alpha_0 = 3
    omega_z = 2*np.pi*1
    omega_p = omega_z/np.sqrt(alpha_0)
    zeta=.4


    a_2 = lambda s: (
        s**2 + 2*zeta*omega_z*s+omega_z**2)/(
        s**2 + 2*zeta*omega_p*s+omega_p**2)
    return a_2


a_2 = get_a_2()

f_hz = .58
print(abs(a_1(complex(0,f_hz*2*np.pi))))
print(abs(a_2(complex(0,f_hz*2*np.pi))))