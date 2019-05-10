import argparse
import os
import cv2
import logging
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import gaussian
import definitions
from gaussian_blur import blur_image, create_gaussian_kernel

def print_with_space(input):
    print('output: {}'.format(input))
    print('--' * 50)


def show_sample_matrix(input_matrix):
    plt.imshow(input_matrix, cmap='gray')
    plt.show()
    plt.close()
    pass


def normalize_gradient(smoothed_gray_image):
    gradient = np.gradient(smoothed_gray_image)
    norms = np.linalg.norm(gradient, axis=0)
    gradient = [np.where(norms == 0, 0, i / norms) for i in gradient]
    return gradient


def find_feature(smoothed_gray_image, input_template_path):
    beta = 900000
    alpha =900000
    contour = np.transpose(input_template_path)
    matrix_A = np.zeros((12, 12))

    # Create A Matrix
    for index in range(12):
        matrix_A[index, index - 2] = beta
        matrix_A[index, index - 1] = -1 * alpha - 4 * beta
        matrix_A[index, index] = 2 * alpha + 6 * beta
        matrix_A[index, index - (np.size(matrix_A, 0) - 1)] = -1 * alpha - 4 * beta
        matrix_A[index, index - (np.size(matrix_A, 0) - 2)] = beta

    gradientX_of_image, gradientY_of_image = np.gradient(smoothed_gray_image)

    gradientXX_of_image, gradientXY_of_image = np.gradient(gradientX_of_image)

    gradientYX_of_image, gradientYY_of_image = np.gradient(gradientY_of_image)
    # show_sample_matrix(gradientX_of_image)
    # show_sample_matrix(gradientY_of_image)
    # show_sample_matrix(gradientXX_of_image)
    # show_sample_matrix(gradientXY_of_image)

    #################### TEST ############################
    # gradientX_of_image, gradientY_of_image = np.gradient(smoothed_gray_image)
    #
    # gradientXX_of_image, gradientXY_of_image = np.gradient(gradientX_of_image)
    #
    # gradientYX_of_image, gradientYY_of_image = np.gradient(gradientY_of_image)
    #
    # show_sample_matrix(gradientX_of_image)

    # Create B terms for current Image
    number_of_contour_points = np.size(contour[0])
    term_Bx = np.zeros(number_of_contour_points)
    term_By = np.zeros(number_of_contour_points)
    # contour = np.transpose(contour)
    # Populate B term with [Ix(x_k,y_k), Iy(x_k,y_k)]*[dIx/dx_k, dIy/dy_k]^T
    for i in range(number_of_contour_points):

        Ix_xkyk = gradientX_of_image[round(contour[1, i]).astype(int), round(contour[0, i]).astype(int)]
        Iy_xkyk = gradientY_of_image[round(contour[1, i]).astype(int), round(contour[0, i]).astype(int)]

        Ixx_xkyk = gradientXX_of_image[round(contour[1, i]).astype(int), round(contour[0, i]).astype(int)]
        Ixy_xkyk = gradientXY_of_image[round(contour[1, i]).astype(int), round(contour[0, i]).astype(int)]

        Iyx_xkyk = gradientYX_of_image[round(contour[1, i]).astype(int), round(contour[0, i]).astype(int)]
        Iyy_xkyk = gradientYY_of_image[round(contour[1, i]).astype(int), round(contour[0, i]).astype(int)]

        term_Bx[i] = np.matmul([Ix_xkyk, Iy_xkyk], np.transpose([Ixx_xkyk, Ixy_xkyk]))
        term_By[i] = np.matmul([Ix_xkyk, Iy_xkyk], np.transpose([Iyx_xkyk, Iyy_xkyk]))

    term_b = np.transpose(np.array([term_Bx, term_By]))
    # contour = np.matmul(np.linalg.inv(matrix_A), term_b)
    # contour = np.transpose(contour)

    # gamma = 1
    # contour = np.matmul(np.linalg.inv(matrix_A + gamma*np.eye(number_of_contour_points)), b_term)

    lam = 900000000
    # print_with_space('contourt[0] : \n{}'.format(contour[0]))
    # contour = np.transpose(contour)

    ####### DIRECT ###########
    contour = np.matmul(np.matmul(np.linalg.inv(matrix_A), np.transpose(matrix_A)) + lam * np.identity(12), term_b)


    # template_contour = contour
    # for i in range(2):
    #     contour = contour - lam * (np.matmul(matrix_A, contour) - np.matmul(matrix_A, template_contour) - term_b_tranpose)
        # print_with_space(contour[0])
        # contour[1] = np.subtract(contour[1], lam * (np.matmul(matrix_A, contour[1] + term_By)))
        # print_with_space(contour[1])
    #
    #
    #     ######################################
    #     contour[0] = np.subtract(contour[0], lam * (np.subtract(np.matmul(matrix_A, contour[0]),
    #         np.matmul(matrix_A, template_contour[0]), term_Bx)))
    #     print_with_space(contour[0])
    #     contour[1] = np.subtract(contour[1], lam * (np.subtract(np.matmul(matrix_A, contour[1]),
    #                                                             np.matmul(matrix_A, template_contour[1]), term_By)))
    #     print_with_space(contour[1])
    #     ######################################

    return np.transpose(contour)


def save_active_contour_image(input_rgb_image, contour_points, output_file_path, debug=False):
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
    # plt.plot(x, y, 'or')  ///Dont think ill need this
    plt.plot(contour_points[0], contour_points[1], '-b')
    if debug:
        plt.show()
    else:
        plt.savefig(output_file_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def main():
    # Setup Log
    logging.basicConfig(filename='active_contour.log', level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')

    # Setup argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--directory", required=True,
                    help="path to data directory")
    ap.add_argument("-r", "--root", required=True, help="Choose root image file name:\n\t"
                                                        "* liptracking2\n\t"
                                                        "* liptracking3\n\t"
                                                        "* liptracking4\n")
    ap.add_argument("-idx1", "--first_index", required=True, help="Choose starting index:\n\t"
                                                                  "01302 - 01910 for liptracking2\n\t"
                                                                  "01295 - 01928 for liptracking3\n\t"
                                                                  "00068 - 00338 for liptracking4\n")
    ap.add_argument("-idx2", "--second_index", required=True, help="Choose ending index: \n\t"
                                                                   "Index ranges from 01302 - 01910")
    ap.add_argument("-t", "--template", required=True, help="Choose Template: \n\t"
                                                            "* lip_tracking2_template\n\t"
                                                            "* lip_tracking3_template\n\t"
                                                            "* lip_tracking4_template\n")
    args = vars(ap.parse_args())

    # Save Parameters
    input_directory = args['directory']
    input_root_file = args['root']
    input_start_index = args['first_index']
    input_last_index = args['second_index']
    input_template = args['template']
    contour_positions = np.load('{}'.format(input_template))

    # First and last image path (Might need later)
    # first_image_path = os.path.join(input_directory, '{}_{}.jpg'.format(input_root_file, input_start_index))
    # last_image_path = os.path.join(input_directory, '{}_{}.jpg'.format(input_root_file, input_last_index))
    first_image_path = '{}_{}.jpg'.format(input_root_file, input_start_index)
    last_image_path = '{}_{}.jpg'.format(input_root_file, input_last_index)

    # Print input parameters to log
    logging.info('Input Parameters:\n'
                 '\tDirectory: {}\n'
                 '\tRoot: {}\n'
                 '\tFirst_Index: {}\n'
                 '\tLast_Index: {}\n'
                 '\tTemplate: {}\n'.format(input_directory, input_root_file, input_start_index, input_last_index,
                                           input_template))

    # Create Gaussian Kernel to be used in smoothing images

    # Create output image folder if doesn't exits
    output_image_folder = os.path.join(os.path.dirname(input_directory), 'output_images')
    if not os.path.exists(output_image_folder):
        os.mkdir(output_image_folder)

    # Will iterate through the directory from start_image to end_image and output image with active contour
    # If sample images in project folder and input_directory is just folder name
    image_directory = os.path.join(definitions.ROOT_DIR, input_directory)
    for image_file_name in sorted(os.listdir(image_directory)):
        if image_file_name >= first_image_path:
            if image_file_name <= last_image_path:
                logging.info('Processing image {}'.format(image_file_name))
                image_file_path = os.path.join(image_directory, image_file_name)
                input_bgr_image = cv2.imread(image_file_path)
                input_rgb_image = cv2.cvtColor(input_bgr_image, cv2.COLOR_BGR2RGB)
                input_gray_image = cv2.cvtColor(input_rgb_image, cv2.COLOR_BGR2GRAY)

                ###########################
                ## Use optimized version!
                # smoothed_gray_image = blur_image(gaussian_kernel, input_gray_image)
                # show_sample_matrix(smoothed_gray_image)

                # smoothed_gray_image_cv2_normalized = filters.gaussian_filter((smoothed_gray_image - smoothed_gray_image.min()) / (
                #             smoothed_gray_image.max() - smoothed_gray_image.min()), 10)

                smoothed_gray_image = gaussian(input_gray_image, 2)
                # show_sample_matrix(smoothed_gray_image)
                ###########################
                # Not sure if smoothed image should be normalized

                # Calculate contour positions
                contour_positions = find_feature(smoothed_gray_image, contour_positions)

                output_file_name = "{}/{}.png".format(output_image_folder, image_file_name[0:-4])
                # save_active_contour_image(input_rgb_image, contour_positions, output_file_name)
                save_active_contour_image(input_rgb_image, contour_positions, output_file_name, debug=True)
    logging.info('Pictures saved in {}'.format(output_image_folder))


if __name__ == '__main__':
    main()
