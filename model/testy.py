
import numpy as np
import matplotlib.pyplot as plt
import cv2
#import scipy.signal as conv



def dijkstra(thresholded_grad, C, ridge_start_row, ridge_start_col):
    m, n = thresholded_grad.shape
    distance_matrix = np.full((m, n), np.inf)
    distance_matrix[ridge_start_row, ridge_start_col] = 0
    visited = np.zeros((m, n))
    visited[ridge_start_row, ridge_start_col] = 1
    edge = C - thresholded_grad
    row = ridge_start_row
    col = ridge_start_col
    previous_pixel = np.empty([m, n], dtype=object)
    previous_pixel[row][col] = [row, col]
    while (0 in visited):
        up = row - 1 if row >= 1 else 0
        down = row + 1 if row < m - 1 else m - 1
        left = col - 1 if col >= 1 else 0
        right = col + 1 if col < n - 1 else n - 1
        print(up, down, left, right, row, col)
        if ((up >= 0) & (visited[up, col] != 1)):
            updistance = edge[up, col]
            if ((distance_matrix[row, col] + updistance) < distance_matrix[up, col]):
                distance_matrix[up, col] = distance_matrix[row, col] + updistance

        if ((left >= 0) & (visited[row, left] != 1)):
            leftdistance = edge[row, left]
            if ((distance_matrix[row, col] + leftdistance) < distance_matrix[row, left]):
                distance_matrix[row, left] = distance_matrix[row, col] + leftdistance

        if ((down < m) & (visited[down, col] != 1)):
            downdistance = edge[down, col]
            if ((distance_matrix[row, col] + downdistance) < distance_matrix[down, col]):
                distance_matrix[down, col] = distance_matrix[row, col] + downdistance
        if ((right < n) & (visited[row, right] != 1)):
            rightdistance = edge[row, right]
            if ((distance_matrix[row, col] + rightdistance) < distance_matrix[row, right]):
                distance_matrix[row, right] = distance_matrix[row, col] + rightdistance

        edges = [updistance, downdistance, leftdistance, rightdistance]

        if (updistance == min(edges)):
            visited[up, col] = 1
            previous_pixel[up][col] = [row, col]
            row = up
        elif (downdistance == min(edges)):
            visited[down, col] = 1
            previous_pixel[down][col] = [row, col]
            row = down
        elif (leftdistance == min(edges)):
            visited[row, left] = 1
            previous_pixel[row][left] = [row, col]
            col = left
        elif (rightdistance == min(edges)):
            visited[row, right] = 1
            previous_pixel[row][right] = [row, col]
            col = right
    # Implement your function here

    return distance_matrix, previous_pixel


plt.rcParams['figure.figsize'] = (15, 15)
plt.rcParams['image.cmap'] = 'gray'

fig = plt.figure(figsize=(8, 6))
img = cv2.imread('images/mountain.png',0)
img_filter = img
grad_x = cv2.Sobel(img_filter,cv2.CV_64F,1,0,ksize=3)
grad_y = cv2.Sobel(img_filter,cv2.CV_64F,0,1,ksize=3)
grad_mag = np.sqrt(np.power(grad_x,2)+np.power(grad_y,2))
th_grad_img = np.copy(grad_mag)
th_grad_img[th_grad_img<80]=0
grady = plt.imshow(th_grad_img,cmap="jet")
plt.colorbar(fraction=0.046, pad=0.04)

ridge_start_row = 67;
ridge_start_col = 15;
C = np.max(th_grad_img)

dist_matrix, prev_pxl = dijkstra(th_grad_img, C, ridge_start_row, ridge_start_col)
fig, ax = plt.subplots()
img1 = ax.imshow(dist_matrix, cmap='jet')
fig.colorbar(img1, ax=ax)
ax.set_aspect('auto')


