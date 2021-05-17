import matplotlib.pyplot as plt
import numpy as np
from play_with_hdf5 import fix_overphase, angle
from uniform_style import *

freqs = np.logspace(-1,1,10000)
s = complex(0, 1)*freqs*2*np.pi
j_m = 50
b_m = 0.01
k_1 = 1000
b_1 = 100
dt = 0.001

K0=2

slide = .5

omega_zeros = 2*np.pi*slide
omega_poles = omega_zeros/np.sqrt(K0+1)
zeta_poles = .4
zeta_zeros = .4




omega_torque = 2*np.pi*10
zeta_torque = .4
C_E = lambda s: 1/(j_m*s*s+b_m*s) # exoskeleton's open loop compliance (also environment side compliance)
torque_bandwidth = lambda s: omega_torque**2/(s*s+2*zeta_torque*omega_torque*s+omega_torque**2)
delay = lambda s: np.exp(-dt*s)
C_A = lambda s: C_E(s)*delay(s)*torque_bandwidth(s) # exo compliance with respect to desired acutator torques
K = lambda s: (s**2+2*zeta_zeros*omega_zeros*s+omega_zeros**2)/(s**2 + 2*zeta_poles*omega_poles*s+omega_poles**2)- 1.0


virtual_series = lambda s: C_A(s)*K(s)
C_H = lambda s: C_E(s)+ virtual_series(s) # human-side compliance (of exo)
alpha = lambda s: C_H(s)/C_E(s)
H = lambda s: complex(4e-4, -3e-4)+0.0*s # human's compliance
tot_compliance = lambda s: 1./(1./H(s)+1./C_H(s))

fig, (ax1, ax2) = plt.subplots(2,1,sharex=True,figsize=(3.5,3.5))

unity = lambda s: C_E(s)/C_E(s)
feedback = lambda s: virtual_series(s)/C_E(s)
amp = lambda s: C_H(s)/C_E(s)

def plot(tf, label, **kwargs):
	ax1.loglog(freqs, abs(tf(s)), **kwargs)
	ax2.semilogx(*fix_overphase(freqs, 180/np.pi*np.angle(tf(s))), label=label, **kwargs)


# Open loop
# figname = "exp_amp_open_loop.pdf"
# plot(C_E, r"\$C_E(s)\$")
# plot(C_A, r"\$C_A(s)\$")

# Amplification Controller
# figname = "exp_amp_amplificiation_controller.pdf"
# plot(unity, "unity")
# plot(K, r"\$K(s)\$")
# plot(alpha, r"\$\alpha(s)\$")

# Realized Behavior
# figname = "exp_amp_realized.pdf"
# plot(C_E, r"\$C_E(s)\$")
# plot(C_H, r"\$C_H(s)\$")
# plot(H, r"\$H(s)\$")

# Effect on Reflection of Human
figname = "exp_amp_effect_on_human.pdf"
plot(C_E, r"\$C_E(s)\$")
plot(lambda s: H(s)/alpha(s), r"\$\frac{H(s)}{\alpha(s)\$")
plot(lambda s: 1/(1/C_E(s) + alpha(s)/H(s)), r"res")

# plot(feedback, "control feedback")
# plot(amp, "amplification")
# ax1.loglog(freqs, abs(unity(s)))
# ax2.semilogx(*fix_overphase(freqs, 180/np.pi*np.angle(unity(s))), label="unity")
# ax1.loglog(freqs, abs(feedback(s)))
# ax2.semilogx(*fix_overphase(freqs, 180/np.pi*np.angle(feedback(s))), label="control feedback")
# ax2.semilogx(*fix_overphase(freqs, 180/np.pi*np.angle(-hum_compliance(s))), "#AAAAAA")
# ax1.loglog(freqs, abs(amp(s)))
# ax2.semilogx(*fix_overphase(freqs, 180/np.pi*np.angle(amp(s))), label="amplification")
# ax1.loglog(freqs, abs(hum_compliance(s)))
# ax2.semilogx(*fix_overphase(freqs, 180/np.pi*np.angle(hum_compliance(s))), label="human")
# ax1.loglog(freqs, abs(tot_compliance(s)), "#444444")
# ax2.semilogx(*fix_overphase(freqs, 180/np.pi*np.angle(tot_compliance(s))), "#444444", label="system")
ax2.grid(True)
plt.legend()
ax1.grid(True)
ax2.set_yticks([-180, -90, 0, 90, 180])
plt.tight_layout()
plt.savefig(figname)
plt.show()

