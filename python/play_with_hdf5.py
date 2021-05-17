# Gray Thomas, NSTRF15AQ33H at NASA JSC August 2018
# Works in python 3
import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from uniform_style import *

font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 9}

matplotlib.rc('font', **font)
# f = h5py.File('2018_08_06_183828_test_logger[-0.0,-0.0,-0.0,-0.0].hdf5')
# open('/home/gray/hdf5_logs/2018_08_06_184256_test_logger.hdf5', 'r')
# f = h5py.File('/home/gray/hdf5_logs/2018_08_06_184256_test_logger.hdf5')
sample_rate = 1000.0
def fft_plot(signal):
    sample_rate = 1000.0
    A = np.fft.rfft(signal[:,1])/(sample_rate)
    f = np.fft.rfftfreq(len(signal), d=1./sample_rate)
    return A, f



def fix_overphase(freqs, phase):
    nfreqs = []
    nphase = []
    op=phase[0] # oldphase
    for f, p in zip(freqs, phase):
        if abs(op-p)>89:
            nfreqs.append(f)
            nphase.append(float("nan"))
        nfreqs.append(f)
        nphase.append(p)
        op=p
    return nfreqs, nphase


def angle(n):
    ang = [a if a<=1e-6 else a-2*np.pi for a in np.angle(n)]
    return np.array(ang)


def plot_discrete(data):
    print(data)
    time = [0.0]
    value = [data[0,1]]
    time0 = data[0,0]
    for t, v in data[1:]:
        time.append(t-time0)
        time.append(t-time0)
        value.append(value[-1])
        value.append(v)
    print(len(time), len(value))
    return time, value

def plot_discrete_autoscaled(data, maintain_zero=True, sign=1):
    print(data)
    time = [0.0]
    maxv = max(data[:,1])
    minv = min(data[:,1])
    scale = sign*1./(maxv-minv)
    mid = (maxv+minv)*0.5
    if (maintain_zero): mid=0
    time0 = data[0,0]
    value = [scale*(data[0,1]-mid)]
    for t, v in data[1:]:
        time.append(t-time0)
        time.append(t-time0)
        value.append(value[-1])
        value.append(scale*(v-mid))
    print(len(time), len(value))
    return time, value

def average_tf(ins, n=10):
    # this doesn't seem to help
    kern = np.ones((n,))/n
    outs = []
    for ith_input in ins:
        outs.append(np.convolve(ith_input[1:], kern))
    return outs

def get_cross_and_power_raw(signalA, signalB):
    fftA, f = fft_plot(signalA)
    fftB, fB = fft_plot(signalB)
    assert len(f)==len(fB)
    return fftA*fftB.conjugate(), (fftB*fftB.conjugate()), f


def hann_window(M):
    x = np.linspace(-.5, .5, M)
    y = .5*np.cos(2*np.pi*x)
    return y

def get_cross_and_power_hann(signalA, signalB, window_size=1024*6, window_number=12):
    # Now with overlapping Hann windows
    window = hann_window(window_size)
    total_length = len(signalA)
    active_length = total_length-window_size
    print ("nominal window number", active_length//window_size)
    window_number*=active_length//window_size
    increment = active_length//(window_number-1)
    sample_rate = 1000.0
    f = np.fft.rfftfreq(window_size, d=1./sample_rate)
    cross, power = complex(0,0)*np.zeros((len(f),)), complex(0,0)*np.zeros((len(f),))
    for n in range(window_number):
        A = np.fft.rfft(window*signalA[n*increment:(n*increment+window_size),1])/(sample_rate)
        B = np.fft.rfft(window*signalB[n*increment:(n*increment+window_size),1])/(sample_rate)
        cross+=A*B.conjugate()
        power+=B*B.conjugate()
    return cross, power, f

def get_cross_and_power_hann2(signalA, signalB, window_size=1024, window_number=120, weighting_power=0.0):
    # special cases of weighting power:
    #   0.0 --- weighted by spectral power
    #   0.5 --- unweighted (standard averaging, corrected for phase of input)
    #   1.0 --- average of trasfer function estimates
    #   1.5 --- weighted by inverse of spectral power
    # Now with overlapping Hann windows
    window = hann_window(window_size)
    total_length = len(signalA)
    active_length = total_length-window_size
    print ("nominal window number", active_length//window_size)
    window_number*=active_length//window_size
    increment = active_length//(window_number-1)
    sample_rate = 1000.0
    f = np.fft.rfftfreq(window_size, d=1./sample_rate)
    cross, power = complex(0,0)*np.zeros((len(f),)), complex(0,0)*np.zeros((len(f),))
    for n in range(window_number):
        A = np.fft.rfft(window*signalA[n*increment:(n*increment+window_size),1])/(sample_rate)
        B = np.fft.rfft(window*signalB[n*increment:(n*increment+window_size),1])/(sample_rate)
        cross+=A*B.conjugate()/((B.real**2+B.imag**2)**weighting_power)
        power+=B*B.conjugate()/((B.real**2+B.imag**2)**weighting_power)
    return cross, power, f

def hann_phasor(signal, reference, window_size=1024, window_number=120, **kwargs):
    # weighting power:
    #   0.5 --- unweighted (standard averaging, corrected for phase of input)
    window = hann_window(window_size)
    total_length = len(signal)
    active_length = total_length-window_size
    print ("nominal window number", active_length//window_size)
    window_number*=active_length//window_size
    increment = active_length//(window_number-1)
    sample_rate = 1000.0
    f = np.fft.rfftfreq(window_size, d=1./sample_rate)
    phasor = complex(0,0)*np.zeros((len(f),))
    for n in range(window_number):
        A = np.fft.rfft(window*signal[n*increment:(n*increment+window_size),1])/(sample_rate)
        B = np.fft.rfft(window*reference[n*increment:(n*increment+window_size),1])/(sample_rate)
        phasor+=A*B.conjugate()/np.abs(B)
    return phasor, f

def tf_A_over_B(signalA, signalB, eps=(1e4, 1e8), ax=plt):
    # fftA, f = fft_plot(signalA)
    # fftB, fB = fft_plot(signalB)
    # assert len(f)==len(fB)
    # transfer = fftA*fftB.conjugate() / (fftB*fftB.conjugate())
    cross, power, f = get_cross_and_power(signalA, signalB)
    transfer = cross / power
    scatter_bode_mag(f, transfer, power, eps=eps, ax=ax)

def scatter_bode_mag(f, transfer, power, eps=(1e4, 1e8), ax=plt):
    normalizer = colors.Normalize(vmin=np.log(eps[0]), vmax=np.log(eps[1]), clip=False)
    input_power = normalizer(np.log(np.abs(power)))
    # transfer = np.abs(fftA) / np.abs(fftB)
    ax.loglog(f, abs(transfer), alpha=0.5)
    ax.scatter(f, abs(transfer), marker='.', c=input_power, cmap="binary")

def scatter_bode_mag(f, transfer, power, eps=(1e4, 1e8), ax=plt):
    normalizer = colors.Normalize(vmin=np.log(eps[0]), vmax=np.log(eps[1]), clip=False)
    input_power = normalizer(np.log(np.abs(power)))
    # transfer = np.abs(fftA) / np.abs(fftB)
    ax.loglog(f, abs(transfer), alpha=0.5)
    ax.loglog(f, abs(transfer), zorder=10, marker='.')

def scatter_bode_mag(f, transfer, ax=plt, **kwargs):
    # ax.loglog(f, abs(transfer), alpha=0.0)
    ax.loglog(f, abs(transfer), marker='.', lw=0.0, **kwargs)

def scatter_bode_phase(f, transfer, ax=plt, **kwargs):
    ax.semilogx(f, 180./np.pi*angle(transfer), marker='.', lw=0.0, **kwargs)

    if ax is plt:
        ax.yticks([-360,-270,-180, -90, 0])
        ax.gca().yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(30))
    else:
        ax.set_yticks([-360,-270,-180, -90, 0])
        ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(30))

def tf_A_over_Bf_split(signalA, signalB, split=500, eps=(1e4, 1e8)):
    n_splits = int(len(signalA)/split)
    cross, power = 0., 0.
    for i in range(n_splits):
        fftA, f = fft_plot(signalA[i*split:(i+1)*split])
        fftB, fB = fft_plot(signalB[i*split:(i+1)*split])
        cross = (fftA*fftB.conjugate())[1:] + cross
        power = (fftB*fftB.conjugate())[1:] + power
        
        assert len(f)==len(fB)
    transfer = cross/power
    normalizer = colors.Normalize(vmin=np.log(eps[0]), vmax=np.log(eps[1]), clip=False)
    input_power = normalizer(np.log(np.abs(power)))
    # transfer = np.abs(fftA) / np.abs(fftB)
    plt.loglog(f[1:], abs(transfer), lw=3, alpha=.8)
    # plt.scatter(f[1:], abs(transfer), marker='.', c=input_power, cmap="bone")

def tf_A_over_B_phase(signalA, signalB, eps=(1e4, 1e8)):
    cross, power, f = get_cross_and_power(signalA, signalB)
    transfer = cross / power
    scatter_bode_phase(f, transfer, power, eps=eps)

def tf_A_over_Bf_split_phase(signalA, signalB, split=500, eps=(1e4, 1e8)):
    n_splits = int(len(signalA)/split)
    cross, power = 0., 0.
    for i in range(n_splits):
        fftA, f = fft_plot(signalA[i*split:(i+1)*split])
        fftB, fB = fft_plot(signalB[i*split:(i+1)*split])
        cross = fftA*fftB.conjugate() + cross
        power = fftB*fftB.conjugate() + power
        
        assert len(f)==len(fB)
    transfer = cross/power
    normalizer = colors.Normalize(vmin=np.log(eps[0]), vmax=np.log(eps[1]), clip=False)
    input_power = normalizer(np.log(np.abs(power)))
    # transfer = np.abs(fftA) / np.abs(fftB)
    plt.semilogx(f[1:], 180./np.pi*angle(transfer[1:]), lw=3, alpha=.8)
    # plt.scatter(f, 180./np.pi*angle(transfer), marker='.', c=input_power, cmap="bone")
    plt.yticks([-360,-270,-180, -90, 0])
    plt.gca().yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(30))
