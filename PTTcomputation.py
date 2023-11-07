import numpy as np
import math
import statsmodels.api as sm
from scipy.signal import find_peaks

def window_rms(a, window_size):
    # Funzione che calcola il window RMS di una funzione 
    # a: segnale su cui calcolare il window RMS
    # window_size: dimensione della finestra mobile 
    # rms: vettore delle stesse dimensioni di a, in cui ogni elemento è calcolato come il valore rms del segnale finestrato, con finestra centrata sul campione di interesse
    
    a2 = [pow(aa,2) for aa in a]
    window = np.ones(int(window_size)) / float(window_size)  # creo array di lunghezza pari
    rms = (np.sqrt(np.convolve(a2, window,'same')))
    return rms #ottengo un vettore delle stesse dimensioni del segnale ma con valori ai bordi approssimati


def vPWV_TD_percentage( x, fs ):
    # This function computes the time-domain envelope of the Doppler-shift signal
    # and identifies the PW footprint as the 5% of the peak amplitude
    # Inputs:
    # x = doppler-shift vector of length T sec [Volts]
    # fs = sampling frequency [Hz]
    # Outputs:
    # latency = scalar [sec]
    # v = velocitogram profile [normalized units]
    
    # Parameters
    T = 0.5                 # length of input signal [sec]
    span = 0.1              # percentage of signal length to be used with the smoothing function
    bs = 0.05 * fs * T      # window at the beginning of the signal in which I assume no PW is present
    MPW = 0.1 * fs * T      # minimum peak width
    th = 5                  # threshold as percentage of peak amplitude
    Pxx_1k_threshold = 0  # threshold value estimated by PW_database_inflating_pressure

    # Check the 0.1-1kHz average power
    Sxx = np.fft.rfft( x )
    Pxx = 20 * np.log10( np.abs(Sxx) + 1e-12 )
    Pxx_1k = sum( Pxx[100:1000] ) / 1000

    if Pxx_1k > Pxx_1k_threshold:

        # Envelope computation
        dx = np.diff(x)              # first derivative of doppler-shift
        v = window_rms( dx, fs/100); # first derivative envelope
        v = np.append(v, v[-1])      # padding to match len(x), duplicating the last sample

        # Smoothing
        points = np.arange(0, len(v), 1)                                                # support vector used for smoothing
        smooth = sm.nonparametric.lowess( v, points, frac=span, it=0, is_sorted=True )    # non paramteric smoothing
        v = smooth[:, 1]
        
        # Normalization
        v = v - np.min(v)
        v = v / np.max(v)

        # PW's Peak identification
        [peaks, property] = find_peaks( v[int(bs):], width=math.floor(MPW) )

        # PW's Footprint identification
        if len(peaks) > 0:
            # Peak position
            peak = int(peaks[0] + bs)
            
            # Valley identification
            valley =  np.argmin( v[ int(bs/2) : int(peak) ] ) # identify the local minimum before the peak
            valley = int(bs/2) + valley

            # Threshold is set at 5% of the peak prominence
            H = (v[peak] - v[valley]) / 100 * th + v[valley]  # compute the peak prominence as the height of the peak from the valley
            percentage = np.argmin( abs(v[valley:peak] - H) ) # identify the threshold crossing starting from the valley
            percentage = valley + percentage                  # identify the threshold crossing starting from the beginning of the signal

            # Transform PTT in seconds
            latency = percentage / fs
        else:
            latency = float('NaN')
            v = float('NaN')
    else:
        latency = float('NaN')
        v = float('NaN')

    # Return the Pulse Transit Time and the velocity profile
    return latency, v