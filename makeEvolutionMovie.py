import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from matplotlib.animation import FuncAnimation

plt.rcParams['animation.ffmpeg_path'] = '/Users/simonvanvliet/opt/miniconda3/envs/mls/bin/ffmpeg'

nBinOffsprSize = 100
nBinOffsprFrac = 100    

binsOffsprSize = np.linspace(0, 0.5, nBinOffsprSize+1)
binsOffsprFrac = np.linspace(0, 1, nBinOffsprFrac+1)

binCenterOffsprSize = (binsOffsprSize[1::]+binsOffsprSize[0:-1])/2
binCenterOffsprFrac = (binsOffsprFrac[1::]+binsOffsprFrac[0:-1])/2

alphaTop = 0.7

#cmapbg = matplotlib.cm.get_cmap(name='gist_gray')
#cmapbg.set_bad(color='black')
#
#cmapfr = matplotlib.cm.get_cmap(name='hot')
#cmapfr.set_bad(color='black')
#    
#cmapFR = matplotlib.cm.ScalarMappable(cmap=cmapfr)
#cmapBG = matplotlib.cm.ScalarMappable(cmap=cmapbg)


def process_frame(data):
    processedData = np.copy(data)
    for ff in range(nBinOffsprFrac):
        for ss in range(nBinOffsprSize):
            toLow = binsOffsprFrac[ff] < binsOffsprSize[ss]
            toHigh = binsOffsprFrac[ff] > (1-binsOffsprSize[ss+1])
            isEmpty = data[ff,ss] == 0
            if (toLow or toHigh) and isEmpty:
                processedData[ff, ss] = np.nan
    processedData /= np.nanmax(processedData)            
    return processedData

def plot_heatmap(fig, ax, data, data_bg):
    data = process_frame(data)
    
    cmapbg = matplotlib.cm.get_cmap(name='gist_gray')
    cmapbg.set_bad(color='black')
    
    cmap = matplotlib.cm.get_cmap(name='hot')
    cmap.set_bad(color='black')
    
    if not data_bg is None:
        image = ax.imshow(data_bg, cmap=cmapbg,
                    interpolation='nearest',
                    extent=[0, 1, 0, 1],
                    origin='lower',
                    vmin = 0,
                    vmax = 1,
                    alpha = 1,
                    aspect='auto')
    
    image = ax.imshow(data, cmap=cmap,
                    interpolation='nearest',
                    extent=[0, 1, 0, 1],
                    origin='lower',
                    vmin = 0,
                    vmax = 1,
                    alpha = alphaTop,
                    aspect='auto')
    
    return image

def set_frame(frameIdx, image, dataArray, data_bg):
    data = dataArray[frameIdx, :, :]
    data = process_frame(data)
    
    fig, ax = plt.subplots(1, figsize=(1, 1))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.axis("off")
    imNew = plot_heatmap(fig, ax, data, data_bg)
    a = imNew.get_array()
    plt.close(fig)

    image.set_array(a)
    return image

def create_movie(data, movie_name, data_bg=None, fps=25, size=800):
    numFrames = data.shape[0]
    
    if not data_bg is None:
        data_bg /= np.nanmax(data_bg)

    #create plot
    fig, ax = plt.subplots(1, figsize=(1, 1))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.axis("off")

    # Initialise plot
    image = plot_heatmap(fig, ax, data[0, :, :], data_bg)

    animation = FuncAnimation(fig, set_frame, 
        frames=np.arange(numFrames), 
        fargs=(image, data, data_bg),
        interval=1000 / fps)

    # Save movie
    animation.save(movie_name, writer='ffmpeg', dpi=size)
    return None

#if __name__ == "__main__":
#    test_data = np.random.rand(100, 512, 512) 
#    moviename = 'test.mp4'
#    create_movie(test_data, moviename)
