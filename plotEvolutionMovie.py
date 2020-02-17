import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

plt.rcParams['animation.ffmpeg_path'] = '/Users/simonvanvliet/opt/miniconda3/envs/mls/bin/ffmpeg'



def set_frame(frameIdx, image, data):
    # We want up-to and _including_ the frame'th element
    image.set_array(data[frameIdx, :, :])
    maxValue = np.max(data[frameIdx, :, :])
    image.set_clim(0, maxValue)
    return image

def create_movie(data, movie_name, fps=25, size=800):
    numFrames = data.shape[0]

    #create plot
    fig, ax = plt.subplots(1, figsize=(1, 1))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.axis("off")

    # Initialise plot
    image = ax.imshow(data[0, :, :], vmin=0, vmax=1)

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
