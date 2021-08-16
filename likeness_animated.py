import itertools

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sound_util import load_wav, write_wav
import ffmpeg





### Get data ###

def get_track_data(start_seconds = 10):
    start = int(44100*start_seconds) 
    song_file = '../sounds/songsinmyhead/49sober.wav'
    data = load_wav(song_file)
    data = data[start:]
    return data


def get_sine_data(freqs, seconds=100):
    dsize = rate*seconds
    data = np.zeros((dsize))
    for f in freqs:
        data += np.sin(f*2*np.pi*np.arange(dsize)/rate)
    data /= freqs.size
    return data

def get_random_data(seconds=100):

    ## Just plain random data
    data = 2*np.random.random((44100*seconds))
    data -= 1


    return data

def get_tiled_random_data():
    ## Tiling random data
   
    tile_length = 54
    data = 2*np.random.random((tile_length))
    data -= 1
    data = np.tile(data, 10000)
    return data

def get_rolled_random_data():

    ## Adding random data at a roll
    roll_length = 52
    data = get_random_data()
    data += np.roll(data, 145)
    data /= 2.0
    n_rolls = 3
    for i in range(n_rolls):
        data += np.roll(data, roll_length*i)

    data /= n_rolls
    return data


### Analysis ###

def likeness_analyze(data):

    w_min = 1 # minimum wavelength to consider
    w_max = int(np.min([max_wavelength, data.shape[0]/2])) # maximum wavelength to consider

    n_values = 300 # number of different wavelengths to look at
    w_step = int((w_max - w_min)/n_values)
    
    w_consider = 10000 # how many samples to use in comparison
    d_skip = 100 # skip data at this rate to reduce total data analyzed

    wavelengths = range(w_min, w_max, w_step)
    mse = []
    

    for w in wavelengths:
        w_mse = 0

        r_data = np.roll(data, -w)
        
        # mean absolute error
        w_mse += (np.abs(data[:w_consider:d_skip] - r_data[:w_consider:d_skip])).mean()
        mse.append(w_mse)

    max_mse = 2 # since data ranges from -1 to 1 
    mse = np.array(mse)
    mse /= max_mse
    likeness = 1.0 - mse
    
    return likeness, wavelengths


def fft_analyze(data):
    fft_data = np.fft.rfft(data)
    fft_mag = np.absolute(fft_data)
    fft_mag = fft_mag/(1.1*np.max(fft_mag))

    freqs = np.fft.rfftfreq(data.shape[0], 1./44100)
    
    freqs = freqs[1:] # skip zero (not a freqency)
    fft_mag = fft_mag[1:]

    wavelengths = 1./freqs
    wavelengths = 44100*wavelengths

    fft_mag = fft_mag[wavelengths <= max_wavelength]
    wavelengths = wavelengths[wavelengths <= max_wavelength]
    

    return fft_mag, wavelengths


### Animation ###

def data_gen():
    step = int(44100/fps)
    length = 10000

    for i in itertools.count():

        c_start = i*step

        data_chunk = data[c_start: c_start + length]
        likeness, wl_likeness = likeness_analyze(data_chunk)
        fft_mag, wl_fft_mag = fft_analyze(data_chunk) 
        if i % 100 == 0:
            print('data_gen ', i)
        yield likeness, wl_likeness, fft_mag, wl_fft_mag


def run(data):
    # update the data
    likeness, wl_likeness, fft_mag, wl_fft_mag = data

    # reset plot
    ax.clear()



    plt.ylim([0.8, 1])
    plt.plot(wl_likeness, likeness, color='xkcd:sky blue')
    if use_fft:
        plt.scatter(wl_fft_mag, fft_mag, color="xkcd:seafoam")

    return plt

def init():
    plt.clf()
    plt.title('Likeness analysis')
    plt.xlabel('Wavelength (samples)')
    plt.ylabel('Likeness')
    plt.minorticks_on()
    plt.grid(True, 'both')
    plt.ylim([0, 1])





def main_animate(data, save=False):
    fps = 30
    seconds = 5
    frames = 30*seconds 
    # frames = int(np.floor(data.shape[0]/fps)) # whole track

    ani = animation.FuncAnimation(fig, run, data_gen, save_count=frames, repeat=False)
    if save:
        ani.save('outputs/likeness_animation.mp4', fps=fps)
    else:+++
        plt.show()

def main_static(data, save=False):
    init()
    run(next(data_gen()))
    if save:
        plt.savefig('outputs/likeness.png')
    else:
        plt.show()

rate = 44100
max_wavelength = 1000
fps = 30 # frames per second

fig, ax = plt.subplots()
data = get_track_data()
use_fft = False
animate = True
save = True
if animate:
    main_animate(data, save=save)
else:
    main_static(data, save=save)




