import argparse
from scipy.interpolate import splprep, splev
import os
import cv2


try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    pass
try:
    import numpy as np
except ModuleNotFoundError:
    pass


def display_contour(input_rgb_image, contour_points, args):
    x = contour_points[:, 0]
    y = contour_points[:, 1]
    x = np.r_[x, x[0]]
    y = np.r_[y, y[0]]

    # Unsure if should pass u into splev (linear) or own values (cubic)
    # tck, u = splprep([x, y], s=0, per=True, u=np.linspace(0, 12, 25))
    tck, u = splprep([x, y], s=0, per=True)
    parametric_values = np.linspace(0, 1)
    snake_points = splev(parametric_values, tck)
    # snake_points = splev(u, tck)

    ##### This works but displays top and bottom border
    ax = plt.axes([0, 0, 1, 1], frameon=False)

    # Then we disable our xaxis and yaxis completely. If we just say plt.axis('off'),
    # they are still used in the computation of the image padding.
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Even though our axes (plot region) are set to cover the whole image with [0,0,1,1],
    # by default they leave padding between the plotted data and the frame. We use tigher=True
    # to make sure the data gets scaled to the full extents of the axes.
    plt.autoscale(tight=True)
    plt.axis('off')

    # Plot the data.
    plt.imshow(input_rgb_image)
    plt.plot(snake_points[0], snake_points[1], '-b')
    plt.plot(x, y, 'or')
    plt.show()


def load_or_save_template(args, input_rgb_image):
    if os.path.exists(args['input_template']):
        contour_points = np.load('{}'.format(args['input_template']))
        display_contour(input_rgb_image, contour_points, args)
    else:
        plt.imshow(input_rgb_image), plt.title('Pick 12 intial points')
        plt.xticks([]), plt.yticks([])

        contour_points = np.asarray(plt.ginput(12))
        # contour_points = contour_points.astype(int)
        np.save('{}'.format(args['input_template']), contour_points)
        plt.close()

    return contour_points


def main():
    # Setup argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image_name", required=True,
                    help="path to data directory")
    ap.add_argument("-t", "--input_template", required=True, help="Input template path."
                                                                  "If it doesn't exist, it will create it, otherwise"
                                                                  "it will preview it\n")
    args = vars(ap.parse_args())

    # Loading Picture with cv2 results in bgr colormap, transform to rgb
    input_bgr_image = cv2.imread(args['image_name'])
    input_rgb_image = cv2.cvtColor(input_bgr_image, cv2.COLOR_BGR2RGB)

    load_or_save_template(args, input_rgb_image)


if __name__ == '__main__':
    main()
