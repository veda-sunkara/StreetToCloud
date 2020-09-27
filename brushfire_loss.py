import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.path import Path

def inverse_brushfire_alg(img, polygons, nodes):
    h, w = img.shape[:2]
    x, y = np.meshgrid(np.arange(h), np.arange(w)) # make a canvas with coordinates
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x,y)).T 
    total_points = h * w

    weight_image = np.zeros((h,w))
    for node, polygon in zip(nodes, polygons):
        # import pdb; pdb.set_trace()

        # find all points within polygon nodes
        grid = polygon.contains_points(points)
        mask = grid.reshape(h,w)
        X, Y = np.where(mask == 1)

        n_points = X.shape[0]

        max_dist = 0
        distances = {}
        for x, y in zip(X, Y):
            # find the closest distance from interior pixel to node pixel
            dist = np.min(np.sqrt((y-node[:,0])**2 + (x-node[:,1])**2))
            distances[tuple((x,y))] = dist
            max_dist = max(max_dist, dist)

        # TODO: add non-linear weighting function
        # some exponential based on the max distance
        # import pdb; pdb.set_trace()
        for coord, dist in distances.items():
            # linear function
            distances[coord] = (max_dist - dist + 1e-3) / max_dist # add epsilon to avoid 0 weights

        # 
        for coord, weight in distances.items():
            weight_image[coord[0], coord[1]] = weight # normalize weight by total pixels in polygon
            # weight_image[coord[0], coord[1]] = dist  # normalize weight by total pixels in polygon

    return weight_image

# load an annotation image
img_path = '/home/purri/research/water_dots/Sen1_dataset/NoQC_shrunk/India_806980_NoQC.tif'
img = np.asarray(Image.open(img_path))

xw, yw = np.where(img == 1)  # get all water pixels
water_img = np.zeros(img.shape)
water_img[xw, yw] = 1

# create binary water image
_, threshold = cv2.threshold(water_img, 0.5, 1, cv2.THRESH_BINARY) 
threshold = threshold.astype(np.uint8)

# compute edges
# contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

# get colors
cmap = plt.get_cmap('tab20')

# begin figure
plt.figure(figsize=(8, 8))
plt.subplot(1,3,1)
plt.imshow(water_img); plt.title('Annotation')

plt.subplot(1,3,2)

# get polygons from contours
p_count = 0
nodes = []
polygons = []
for i, cnt in enumerate(contours): 
    area = cv2.contourArea(cnt) 
   
    if area > np.prod(img.shape)*0.0001:  # some annotations may be noise so remove annotations less than 0.01% image size
        p_count += 1
        # Shortlisting the regions based on there area. 
        approx = cv2.approxPolyDP(cnt, 0.00009 * cv2.arcLength(cnt, True), True) # small number makes sure we have accurate polygon
        points = approx[:, 0,:]
        X, Y = points[:, 0], points[:, 1]
        nodes.append(points)

        # TODO: How do I handle overlapping polygons? Order of polygons shouldn't matter
        # Use shapely module?
        # TODO: How to also include the background as a polygon?
        # just find points that are zero or background and then find distance to contour points

        # create a polygon
        points_list = []
        for x, y in zip(X, Y):
            points_list.append(tuple((x, y)))
        polygon = Path(points_list)
        polygons.append(polygon)

        # flip y values (for display only)
        max_h = img.shape[0]
        Y = max_h - Y

        plt.fill(X, Y, c=cmap(i/len(contours)))
plt.title('Polygon Image: {}'.format(p_count))

# do inverse brushfire algorithm
inv_brush_img = inverse_brushfire_alg(img, polygons, nodes)

plt.subplot(1,3,3)
plt.imshow(inv_brush_img); plt.title('Edge weight image')
plt.colorbar()
plt.show()
