import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from matplotlib.animation import FuncAnimation

plt.rcParams['animation.ffmpeg_path'] = '/Users/simonvanvliet/opt/miniconda3/envs/mls/bin/ffmpeg'

nBinOffsprSize = 100
nBinOffsprFrac = 100    

binsOffsprSize = np.linspace(0, 1, nBinOffsprSize+1)
binsOffsprFrac = np.linspace(0, 1, nBinOffsprFrac+1)

binCenterOffsprSize = (binsOffsprSize[1::]+binsOffsprSize[0:-1])/2
binCenterOffsprFrac = (binsOffsprFrac[1::]+binsOffsprFrac[0:-1])/2

def process_frame(data):
    processedData = np.copy(data)
    for ff in range(nBinOffsprFrac):
        for ss in range(nBinOffsprSize):
            toLow = binsOffsprFrac[ff] < binsOffsprSize[ss]
            toHigh = binsOffsprFrac[ff] > (1-binsOffsprSize[ss+1])
            isEmpty = data[ff,ss] == 0
            if (toLow or toHigh) and isEmpty:
                processedData[ff, ss] = np.nan
                
    return processedData

def plot_heatmap(fig, ax, data):
    maxValue = np.max(data)
    data = process_frame(data)
    
    cmap = matplotlib.cm.get_cmap(name='viridis')
    cmap.set_bad(color='black')
    
    image = ax.imshow(data, cmap=cmap,
                    interpolation='nearest',
                    extent=[0, 1, 0, 1],
                    origin='lower',
                    vmin = 0,
                    vmax = maxValue,
                    aspect='auto')

    
    return image

def set_frame(frameIdx, image, dataArray):
    # We want up-to and _including_ the frame'th element
    data = dataArray[frameIdx, :, :]
    maxValue = np.max(data)
    data = process_frame(data)
    image.set_array(data)
    image.set_clim(0, maxValue)
    return image

def create_movie(data, movie_name, fps=25, size=800):
    numFrames = data.shape[0]

    #create plot
    fig, ax = plt.subplots(1, figsize=(1, 0.5))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.axis("off")

    # Initialise plot
    image = plot_heatmap(fig, ax, data[0, :, :])

    animation = FuncAnimation(fig, set_frame, 
        frames=np.arange(numFrames), 
        fargs=(image, data),
        interval=1000 / fps)

    # Save movie
    animation.save(movie_name, writer='ffmpeg', dpi=size)
    return None

if __name__ == "__main__":
    test_data = np.random.rand(100, 512, 512) 
    moviename = 'test.mp4'
    create_movie(test_data, moviename)
