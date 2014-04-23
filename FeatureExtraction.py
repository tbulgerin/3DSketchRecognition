__author__ = 'tbulgerin'

import os
import numpy
import math

from matplotlib.mlab import PCA


def featureExtraction():
    # specify the path to the directory containing the points
    path = '/Users/tbulgerin/Desktop/travis_points/'

    # specify the dimensions of the bounding box
    boundingBox = 64

    # specify the cell and block sizes, each will be n x n x n
    # must be a power of 2
    cellSize = 8
    blockSize = 2

    count = 0
    label = 1
    shouldIncrement = False
    for root, sub, files in os.walk(path):

        for file in files:
            shouldIncrement = True

            # read in the x, y, z coordinates for this particular image
            xyzMatrix = readCoordinates(root + '/' + file)

            # perform PCA analysis and get the 2D array of the data projected into PCA space
            pcaMatrix = PCA(xyzMatrix)
            xyzMatrix = pcaMatrix.Y

            # normalize the x, y, z coordinates to fit within a specified bounding box
            xyzMatrix = normalizeDataPoints(xyzMatrix, boundingBox)

            # perform voxelization to get a three-dimensional matrix
            voxelMatrix = voxelization(xyzMatrix, boundingBox)

            # calculate the x, y, and z gradients
            gradientMatrix = numpy.gradient(voxelMatrix)

            # calculate the histograms of cells based on the gradients
            cellHistograms = calculateHistograms(gradientMatrix, boundingBox, cellSize)

            # concatenate cell histograms into block histograms with overlap
            blocksHistograms = calculateBlocks(cellHistograms, blockSize, cellSize, boundingBox)

            # normalize the vector of each block
            normBlocks = normalizeBlocks(blocksHistograms)

            # concatenate block vectors to form final feature vector
            featureVector = constructFeatureVector(normBlocks)

            # save the feature vector into the features directory
            featurePath = os.getcwd()
            writeFeatureVector(featurePath + '/features/', featureVector, label)

            count += 1

        if shouldIncrement:
            label += 1
            shouldIncrement = False


# read in the 3D data points into a numpy matrix
def readCoordinates(path):
    return numpy.loadtxt(path, usecols=(1, 2, 3))


# normalize each of the x, y, z coordinates between 0 and the dimensions of the bounding box
def normalizeDataPoints(xyzMatrix, boundingBox):
    m_min = numpy.min(xyzMatrix, axis=0)
    m_max = numpy.max(xyzMatrix, axis=0)

    n_min = 0
    n_max = boundingBox - 1

    for row in xyzMatrix:
        for i in range(3):
            numerator = (row[i] - m_min[i]) * (n_max - n_min)
            denominator = m_max[i] - m_min[i]
            n_result = n_min + (numerator / denominator)
            row[i] = n_result

    return xyzMatrix.astype(int)


# perform voxelization on the coordinate matrix
def voxelization(xyzMatrix, boundingBox):
    voxelMatrix = numpy.zeros((boundingBox, boundingBox, boundingBox), dtype=int)

    for row in xyzMatrix:
        x = row[0]
        y = row[1]
        z = row[2]
        voxelMatrix[x][y][z] = 1

    return voxelMatrix


# calculate the histograms for each of the cells
def calculateHistograms(gradientMatrix, boundingBox, cellSize):
    d = boundingBox / cellSize
    numCells = d * d * d
    phi_bins = 18
    theta_bins = 9

    histograms = list()
    h = [[0 for j in range(phi_bins)] for i in range(theta_bins)]

    for i in range(numCells):
        histograms.append(h)

    for i in range(boundingBox):
        for j in range(boundingBox):
            for k in range(boundingBox):
                # get the gradients in the x, y and z directions at this voxel
                x = gradientMatrix[0][i][j][k]
                y = gradientMatrix[1][i][j][k]
                z = gradientMatrix[2][i][j][k]

                # map the indices to a cube
                i_cube = i / cellSize
                j_cube = j / cellSize
                k_cube = k / cellSize

                # get the indices of this cell's histogram
                first = cellSize * cellSize * i_cube
                second = cellSize * j_cube
                index = first + second + k_cube

                # if we have a zero gradient, don't bother binning
                if not(x == 0.0 and y == 0.0 and z == 0.0):
                    performBinning(histograms[index], x, y, z)

    # collapse the two-dimensional histogram into a one dimensional histogram
    histograms_collapsed = list()

    for h in histograms:
        tmp = list()
        for v in h:
            tmp.extend(v)
        histograms_collapsed.append(tmp)

    return histograms_collapsed


# calculate the block vector with overlap
def calculateBlocks(cellHistogram, blockSize, cellSize, boundingBox):
    blocksHistogram = list()

    for i in range(len(cellHistogram)):
        b = list()
        bottomRow = False
        rightColumn = False

        # add the histogram for the current cell
        b.extend(cellHistogram[i])

        if i % boundingBox >= (cellSize * (cellSize-1)):
            bottomRow = True
        if (i+1) % cellSize == 0:
            rightColumn = True

        # add the histogram for the next cell in the x
        if not rightColumn:
            if i+1 < len(cellHistogram):
                b.extend(cellHistogram[i+1])

        # add the histogram for the cell below in the y
        if not bottomRow:
            if i+cellSize < len(cellHistogram):
                b.extend(cellHistogram[i+cellSize])

        # add the histogram for the cell in the z
        if (cellSize*cellSize+i) < len(cellHistogram):
            b.extend(cellHistogram[cellSize*cellSize+i])

        # to make a cube code (blockSize x blockSize x blockSize)
        if not bottomRow and not rightColumn:
            if i+cellSize+1 < len(cellHistogram):
                b.extend(cellHistogram[i+cellSize+1])

        if not rightColumn:
            if (cellSize*cellSize+i+1) < len(cellHistogram):
                b.extend(cellHistogram[cellSize*cellSize+i+1])

        if not bottomRow:
            if (cellSize*cellSize+i+cellSize) < len(cellHistogram):
                b.extend(cellHistogram[cellSize*cellSize+i+cellSize])

        if not bottomRow and not rightColumn:
            if (cellSize*cellSize+i+cellSize) < len(cellHistogram):
                b.extend(cellHistogram[cellSize*cellSize+i+cellSize+1])

        blocksHistogram.append(b)

    return blocksHistogram


# normalize each block vector separately
def normalizeBlocks(blocksHistogram):
    normBlocks = list()

    for block in blocksHistogram:
        normBlocks.append(normalizeVector(block))

    return normBlocks


# construct the final feature vector
def constructFeatureVector(blocks):
    featureVector = list()

    for b in blocks:
        featureVector.extend(b)

    return featureVector


# write the model's feature vector to a file
def writeFeatureVector(path, vector, count):
    if not os.path.exists(path):
        os.makedirs(path)

    featureNumber = 1

    f2 = open(path + 'features.txt', 'a')
    f2.write(str(count) + ' ')
    for item in vector:
        if item != 0:
            f2.write(str(featureNumber) + ':' + str(item) + ' ')
        featureNumber += 1

    f2.write('\n')
    f2.close()


# normalize a vector between zero and one
def normalizeVector(vector):
    normVector = list()
    v_min = min(vector)
    v_max = max(vector)

    if v_min == 0 and v_max == 0:
        return [0 for i in range(len(vector))]

    for x in vector:
        norm_x = (x - v_min) / float((v_max - v_min))
        normVector.append(norm_x)

    return normVector


# convert the cartesian coordinates into spherical coordinates
def convertToSpherical(x, y, z):
    phi = numpy.arctan2(y, x)
    r = math.sqrt(x**2 + y**2 + z**2)
    theta = math.acos(z/r)

    return [r, theta, phi]


# find the appropriate bin based on theta and phi
# and increment the histogram
def performBinning(histogram, x, y, z):
    # assign the number of bins for each angle
    theta_bins = 9
    phi_bins = 18

    # convert cartesian to spherical
    [r, theta, phi] = convertToSpherical(x, y, z)

    # work in degrees for the binning
    theta_degrees = math.degrees(theta)
    phi_degrees = math.degrees(phi)

    # make theta between 0 and 180
    n = int(math.fabs(theta_degrees / 180))
    if theta_degrees < 0:
        theta_degrees = (180*(n+1)) + theta_degrees
    elif theta_degrees > 180:
        theta_degrees = theta_degrees - (180*n)

    # make phi between 0 and 360
    n = int(math.fabs(phi_degrees / 360))
    if phi_degrees < 0:
        phi_degrees = (360*(n+1)) + phi_degrees
    elif phi_degrees > 360:
        phi_degrees = phi_degrees - (360*n)

    # find the index of the correct bin for theta
    bin_width = 180 / theta_bins
    for i in range(theta_bins):
        if theta_degrees >= (i * bin_width) and theta_degrees <= ((i+1) * bin_width):
            theta_index = i

    # find the index of the correct bin for phi
    bin_width = 360 / phi_bins
    for i in range(phi_bins):
        if phi_degrees >= (i * bin_width) and phi_degrees <= ((i+1) * bin_width):
            phi_index = i

    # add 1 to the histogram in that spot
    histogram[theta_index][phi_index] += 1


# print the contents of a two dimensional matrix
def printMatrix(xyzMatrix):
    for row in xyzMatrix:
        print "{}\n".format(row)