import imageio
from os.path import isfile, join
from os import listdir

if __name__ == "__main__":
    MY_PATH = ".\\out\\"
    frames = [f for f in listdir(MY_PATH) if isfile(join(MY_PATH, f)) and f.startswith("plot")]
    images = []
    for frame in frames:
        images.append(imageio.imread(join(MY_PATH, frame)))
    imageio.mimsave("out/movie.gif", images)
