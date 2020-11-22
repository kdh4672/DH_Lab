from torchvision.datasets import folder


def exr_loader(path, ndim=3):
    """
    loads an .exr file as a numpy array
    :param path: path to the file
    :param ndim: number of channels that the image has,
                    if 1 the 'R' channel is taken
                    if 3 the 'R', 'G' and 'B' channels are taken
    :return: np.array containing the .exr image
    """
    import OpenEXR
    import Imath
    import numpy as np

    # read image and its dataWindow to obtain its size
    pic = OpenEXR.InputFile(path)
    dw = pic.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    pt = Imath.PixelType(Imath.PixelType.FLOAT)

    if ndim == 1:
        # transform data to numpy
        channel = np.fromstring(channel=pic.channel('R', pt), dtype=np.float32)
        channel.shape = (size[1], size[0])  # Numpy arrays are (row, col)
        return np.array(channel)
    if ndim == 3:
        # read channels indivudally
        allchannels = []
        for c in ['R', 'G', 'B']:
            # transform data to numpy
            channel = np.fromstring(pic.channel(c, pt), dtype=np.float32)
            channel.shape = (size[1], size[0])
            allchannels.append(channel)

        # create array and transpose dimensions to match numpy style
        return np.array(allchannels).transpose((1, 2, 0))


class ExrData(folder.ImageFolder):
    def __init__(self, *args, **kwargs):
        # add the '.exr' extension to load this type of files
        folder.IMG_EXTENSIONS += ['.exr']

        super(exrData, self).__init__(*args, **kwargs)