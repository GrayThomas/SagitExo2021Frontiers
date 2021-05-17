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

slide = .7

for index, slide in enumerate(np.linspace(.5,.9,3)):

	omega_zeros = 2*np.pi*slide
	omega_poles = omega_zeros/np.sqrt(K0+1)
	zeta_poles = .4
	zeta_zeros = .4




	omega_torque = 2*np.pi*10
	zeta_torque = .4
	base = lambda s: 1/(j_m*s*s+b_m*s)
	torque_bandwidth = lambda s: omega_torque**2/(s*s+2*zeta_torque*omega_torque*s+omega_torque**2)
	delay = lambda s: np.exp(-dt*s)
	filt = lambda s: (s**2+2*zeta_zeros*omega_zeros*s+omega_zeros**2)/(s**2 + 2*zeta_poles*omega_poles*s+omega_poles**2)- 1.0
	virtual_series = lambda s: base(s)*delay(s)*torque_bandwidth(s)*filt(s)
	exo_compliance = lambda s: base(s)+ virtual_series(s)
	hum_compliance = lambda s: complex(4e-4, -3e-4)+0.0*s
	tot_compliance = lambda s: 1./(1./hum_compliance(s)+1./exo_compliance(s))

	fig, (ax1, ax2) = plt.subplots(2,1,sharex=True,figsize=(3.5,3.5))
	# ax1.loglog(freqs, abs(base(s)))
	# ax2.semilogx(*fix_overphase(freqs, 180/np.pi*np.angle(base(s))), label="original")
	# ax1.loglog(freqs, abs(virtual_series(s)))
	# ax2.semilogx(*fix_overphase(freqs, 180/np.pi*np.angle(virtual_series(s))), label="control")
	ax2.semilogx(*fix_overphase(freqs, 180/np.pi*np.angle(-hum_compliance(s))), "#AAAAAA")
	ax1.loglog(freqs, abs(exo_compliance(s)))
	ax2.semilogx(*fix_overphase(freqs, 180/np.pi*np.angle(exo_compliance(s))), label="robot")
	ax1.loglog(freqs, abs(hum_compliance(s)))
	ax2.semilogx(*fix_overphase(freqs, 180/np.pi*np.angle(hum_compliance(s))), label="human")
	ax1.loglog(freqs, abs(tot_compliance(s)), "#444444")
	ax2.semilogx(*fix_overphase(freqs, 180/np.pi*np.angle(tot_compliance(s))), "#444444", label="system")
	ax2.grid(True)
	plt.legend()
	ax1.grid(True)
	ax2.set_yticks([-180, -90, 0, 90, 180])
	plt.tight_layout()
	plt.savefig("fig%d.pdf"%index)
plt.show()

