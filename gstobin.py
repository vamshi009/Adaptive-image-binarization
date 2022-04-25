import numpy as np
from PIL import Image
import PIL
def checkvalid(row, col, x_2d):
    rowmax = x_2d.shape[0]
    colmax = x_2d.shape[1]

    if((row>=0 and row <= (rowmax-1) ) and (col >=0 and col <= (colmax-1))):
        return 1
    else:
        return 0


def get_mean(i, j, x_2d, windowsize):

    sum = 0
    for row in list(range(i-windowsize, i+windowsize+1)):
        for col in list(range(j-windowsize, j+windowsize+1)):
            if(checkvalid(row,col, x_2d)):
                sum = sum + x_2d[row][col]

            
    ans = sum/((2*windowsize + 1)*(2*windowsize + 1))
    return ans

def get_sd(i, j, x_2d, windowsize, meanvalue):

    sum = 0
    for row in list(range(i-windowsize, i+windowsize)):
        for col in list(range(j-windowsize, j+windowsize)):
            if(checkvalid(row,col, x_2d)):
                sum = sum + (x_2d[row][col] - meanvalue)*(x_2d[row][col] - meanvalue)

                                


    avgsum = sum/((2*windowsize + 1)*(2*windowsize + 1))

    return np.sqrt(avgsum)


def greyscale_to_binary(x_2d, windowsize):
    rows = x_2d.shape[0]
    cols = x_2d.shape[1]
    print("Input matrix is ", x_2d)
    ths_matrix = [[0]*cols]*rows
    ths_matrix = np.array(ths_matrix, dtype= np.uint8)
    for i in range(rows):
        for j in range(cols):
            print("Index ", i, " ", j)
            meanvalue = get_mean(i,j, x_2d, windowsize)
            sd = get_sd(i,j, x_2d, windowsize, meanvalue)
            threshold = meanvalue*(1 + 0.5*(sd/128 - 1))
            ths_matrix[i][j] = threshold

    print("threshold matrix is ", ths_matrix)
    bin_image = x_2d < ths_matrix
    img = Image.fromarray(bin_image)
    img.save('img2binary32.png')

    return bin_image



if(__name__=='__main__'):
    rows = 5
    cols = 5
    x = [[0]*cols]*rows
    x = np.array(x, dtype= np.uint8)
    an_image = PIL.Image.open("img2.png")
    image_sequence = an_image.getdata()
    image_array = np.array(image_sequence)

    print(image_array)
    print(image_array.shape)
    image_array = np.reshape(image_array, (240, 850))
    print(image_array.shape)

    #251 x 157 img6
    #391 616 img5
    #390 487 img7
    #627 397 img12
    print(greyscale_to_binary(image_array, 32))