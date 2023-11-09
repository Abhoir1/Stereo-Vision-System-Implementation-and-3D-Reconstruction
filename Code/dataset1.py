import cv2 as cv
import numpy as np
import random
import matplotlib.pyplot as plt

print("-------------------------------------------------------------------------------------------------------")
print("Solving dataset 1")
print("-------------------------------------------------------------------------------------------------------")

def show_image_with_resized_window(image, window_name):
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(window_name, image.shape[1], image.shape[0])
    cv.imshow(window_name, image)

def wait_key_or_close(window_name):
    while True:
        key = cv.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif cv.getWindowProperty(window_name, cv.WND_PROP_VISIBLE) < 1:
            break

    cv.destroyAllWindows()

image1_p = "/home/ab/enpm673/project4/artroom/im0.png"
image2_p = "/home/ab/enpm673/project4/artroom/im1.png"

def feature_extracter(image1_path, image2_path, x=250):

    dataset_1_image_1 = cv.imread(image1_path)
    dataset_1_image_2 = cv.imread(image2_path)

    gray1 = cv.cvtColor(dataset_1_image_1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(dataset_1_image_2, cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    image_with_keypoints1 = cv.drawKeypoints(dataset_1_image_1, keypoints1, None)
    image_with_keypoints2 = cv.drawKeypoints(dataset_1_image_2, keypoints2, None)

    features_both_images = np.concatenate((image_with_keypoints1, image_with_keypoints2), axis=1)

    show_image_with_resized_window(features_both_images, "detected features in both the images")
    wait_key_or_close("detected features in both the images")

    cv.imwrite("detected features.png", features_both_images)

    flann_matcher = cv.BFMatcher()
    matches = flann_matcher.match(descriptors1,descriptors2)

    sorted_matches = sorted(matches, key=lambda x:x.distance)

    select_matches = sorted_matches[0:x]

    matched_features = cv.drawMatches(dataset_1_image_1,keypoints1,dataset_1_image_2,keypoints2,sorted_matches[:60],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    show_image_with_resized_window(matched_features, "Matched features between both the images")
    wait_key_or_close("Matched features between both the images")

    cv.imwrite("Matched features.png", matched_features)

    best_matches = [[keypoints1[m.queryIdx].pt[0], keypoints1[m.queryIdx].pt[1], keypoints2[m.trainIdx].pt[0], keypoints2[m.trainIdx].pt[1]] for m in select_matches]
    best_matches = np.array(best_matches).reshape(-1, 4)

    return best_matches, dataset_1_image_1, dataset_1_image_2, gray1, gray2


best_matches, dataset_1_image_1, dataset_1_image_2, gray1, gray2 = feature_extracter(image1_p, image2_p, 250)

def normalizing_function(value):

    mean = np.mean(value, axis=0)

    euclidean_distance = np.sqrt(np.sum((value - mean)**2, axis=1))
    scale = np.sqrt(2) / np.mean(euclidean_distance)

    Translation = np.array([[scale, 0, -scale * mean[0]],
                  [0, scale, -scale * mean[1]],
                  [0, 0, 1]])
    
    value_homo = np.column_stack((value, np.ones((len(value), 1))))
    value_norm = np.dot(Translation, value_homo.T).T

    normalised_value = value_norm[:, :2] / value_norm[:, 2:]

    return normalised_value, Translation


def compute_fundamental_matrix(matches):

    image1_features = matches[:, 0:2]
    image2_features = matches[:, 2:4]
        
    normalised_value1, translation1 = normalizing_function(image1_features)
    normalised_value2, translation2 = normalizing_function(image2_features)

    A = np.zeros((normalised_value1.shape[0], 9))

    for i in range(0, len(normalised_value1)):
            x1,y1 = normalised_value1[i][:2]
            x2,y2 = normalised_value2[i][:2]
            A[i] = np.array([x1*x2, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]) 

    U, S, Vt = np.linalg.svd(A)  
    F = Vt.T[:, -1].reshape(3,3)

    u, s, vt = np.linalg.svd(F)
    s[-1] = 0
    F = u @ np.diag(s) @ vt

    Fundamental_matrix = translation2.T @ F @ translation1

    return Fundamental_matrix

def ransac_fundamental_matrix(matched_features, num_iterations, threshold_distance  ):
     
    best_num_inliners = 0
    best_inliers = []
    best_F_matrix = None

    for i in range(num_iterations):
         
        inliers = []
        indices = random.sample(range(matched_features.shape[0]), 8)
        random_points = matched_features[indices, :]

        fundamental_matrix = compute_fundamental_matrix(random_points)

        for j in range(matched_features.shape[0]):

            temp_feature = matched_features[j]

            x1 = temp_feature[0:2]
            x2 = temp_feature[2:4]

            array1 = np.array([x1[0], x1[1], 1]).T
            array2 = np.array([x2[0], x2[1], 1])

            error = array1 @ fundamental_matrix @ array2
            error = np.abs(error)

            if error < threshold_distance:
              inliers.append(j)

        if len(inliers) > best_num_inliners:
            best_num_inliners = len(inliers)
            best_inliers = inliers
            best_F_matrix = fundamental_matrix

    inlier_points = matched_features[best_inliers, :]

    return best_F_matrix, inlier_points

num_iterations = 1000
threshold_distance = 0.1

best_F_matrix, best_inliers_points = ransac_fundamental_matrix(best_matches, num_iterations, threshold_distance)

print("\n")
print("Fundamental matrix = \n ", best_F_matrix)
print("\n")


def triangulation(points1, points2, projection_matrix_l, projection_matrix_r):

     points1_homo = cv.convertPointsToHomogeneous(points1)
     points2_homo = cv.convertPointsToHomogeneous(points2)

     points1_homo = points1_homo.reshape(2, -1)
     points2_homo = points2_homo.reshape(2, -1)

     points3D_homo = cv.triangulatePoints(projection_matrix_r, projection_matrix_l, points1_homo, points2_homo)

     points3D_nonhomo = cv.convertPointsFromHomogeneous(points3D_homo.T)

     return points3D_nonhomo


best_points_calc1 = best_inliers_points[:,0:2]
best_points_calc2 = best_inliers_points[:,2:4]

cam0 = np.array([[1733.74, 0, 792.27],
                 [0, 1733.74, 541.89],
                 [0, 0, 1]])

cam1 = np.array([[1733.74, 0, 792.27],
                 [0, 1733.74, 541.89],
                 [0, 0, 1]])

K = (cam0 + cam1)/2

E = K.T @ best_F_matrix @ K

U, S, Vt = np.linalg.svd(E)

E = U @ np.diag([1, 1, 0]) @ Vt.T

print("\n")
print("Essential matrix = \n ", E)
print("\n")

U, S, Vt = np.linalg.svd(E)

W = np.array([[0, -1, 0 ],
              [1, 0, 0],
              [0, 0, 1]])

def correct_values(C, R):
     if np.linalg.det(R) < 0:
          R = -R
          C = -C
     return C, R

C1 = U[:, 2]
R1 = U @ W @ Vt
C1, R1 = correct_values(C1, R1)

C2 = -U[:, 2]
R2 = U @ W @ Vt
C2, R2 = correct_values(C2, R2)

C3 = U[:, 2]
R3 = U @ W.T @ Vt
C3, R3 = correct_values(C3, R3)

C4 = -U[:, 2]
R4 = U @ W.T @ Vt
C4, R4 = correct_values(C4, R4)

positive_translations = []

if C1[2] > 0:
     positive_translations.append(C1)
if C2[2] > 0:
     positive_translations.append(C2)
if C3[2] > 0:
     positive_translations.append(C3)
if C4[2] > 0:
     positive_translations.append(C4)

C1_new = positive_translations[0]

R1_new = R1
R2_new = R3

Projectionmatrix_left = np.hstack((cam0, np.zeros((3, 1))))

Projectionmatrix_right1 = np.hstack((cam1 @ R1_new, cam1 @ C1_new.reshape(3, 1)))
Projectionmatrix_right2 = np.hstack((cam1 @ R2_new, cam1 @ C1_new.reshape(3, 1)))

x = int(len(best_points_calc1) + (len(best_points_calc1)/2))

possible3d_1 = (triangulation(best_points_calc1, best_points_calc2, Projectionmatrix_left, Projectionmatrix_right1)).reshape(x, 3)
possible3d_2 = (triangulation(best_points_calc1, best_points_calc2, Projectionmatrix_left, Projectionmatrix_right2)).reshape(x, 3)

possible3d_1_inliers = np.sum(possible3d_1[:, 2] > 0)
possible3d_2_inliers = np.sum(possible3d_2[:, 2] > 0)

index_best_pose = np.argmax([possible3d_1_inliers, possible3d_2_inliers])

if index_best_pose == 0:
     Final_rotation = R1_new
     Final_translation = C1_new
     Final_projection_matrix = Projectionmatrix_right1
elif index_best_pose == 1:
     Final_rotation = R2_new
     Final_translation = C1_new
     Final_projection_matrix = Projectionmatrix_right2

print("\n")
print("Final rotaion matrix = \n ", Final_rotation)
print("\n")
print("Final translation matrix = \n ", Final_translation)
print("\n")

def drawlines_rectified(img1src, img2src, lines, pts1src, pts2src):
    
    img1color = cv.cvtColor(img1src, cv.COLOR_GRAY2BGR)
    img2color = cv.cvtColor(img2src, cv.COLOR_GRAY2BGR)

    np.random.seed(0)

    r, c = img1src.shape

    for r, pt1, pt2 in zip(lines, pts1src, pts2src):

        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, pt1[1]])
        x1, y1 = map(int, [c, pt1[1]])
        img1color = cv.line(img1color, (x0, y0), (x1, y1), color, 3)
        img1color = cv.circle((img1color), (int(pt1[0]), int(pt1[1])), 7, (0, 0, 200), -1)
        
        x0, y0 = map(int, [0, pt2[1]])
        x1, y1 = map(int, [c, pt2[1]])
        img2color = cv.line((img2color), (x0, y0), (x1, y1), (color), 3)
        img2color = cv.circle(img2color, (int(pt2[0]), int(pt2[1])), 7, (0, 0, 200), -1)

    return img1color, img2color

def rectification(imag1, imag2):
    h1, w1 = imag1.shape[:2]
    h2, w2 = imag2.shape[:2]

    retval, H1, H2 = cv.stereoRectifyUncalibrated(np.float32(best_points_calc1), np.float32(best_points_calc2), best_F_matrix, imgSize=(w1, h1))

    print("\n")
    print("Homography matrix for left image  = \n", H1)
    print("\n")
    print("Homography matrix for right image  = \n", H1)
    print("\n")

    return H1, H2, (w1, h1), (w2, h2)

H1, H2, image1_size, image2_size = rectification(dataset_1_image_1, dataset_1_image_2)

def rectified_epilines(imag1, imag2):

    img1_rectified = cv.warpPerspective(imag1, H1, image1_size)
    img2_rectified = cv.warpPerspective(imag2, H2, image2_size)

    points1_rectified = cv.perspectiveTransform(best_points_calc1.reshape(-1, 1, 2), H1).reshape(-1,2)
    points2_rectified = cv.perspectiveTransform(best_points_calc2.reshape(-1, 1, 2), H2).reshape(-1,2)

    lines1 = cv.computeCorrespondEpilines(
        points2_rectified.reshape(-1, 1, 2), 2, best_F_matrix)
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = drawlines_rectified(img1_rectified, img2_rectified, lines1, points1_rectified, points2_rectified)

    lines2 = cv.computeCorrespondEpilines(
        points1_rectified.reshape(-1, 1, 2), 1, best_F_matrix)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawlines_rectified(img2_rectified, img1_rectified, lines2, points2_rectified, points1_rectified)

    result = np.concatenate((img3, img5), axis=1)
    horizontal_concat = cv.hconcat([img1_rectified, img2_rectified])

    return result, horizontal_concat, img1_rectified, img2_rectified

result, horizontal_concat, img1_rectified, img2_rectified = rectified_epilines(gray1, gray2)

show_image_with_resized_window(result, "epilines")
wait_key_or_close("epilines")

cv.imwrite("epilines.png", result)

def drawlines(img1, img2, lines, pts1, pts2):
    r, c = img1.shape

    color1 = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
    color2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
 
    np.random.seed(0)
    for r, pt1, pt2 in zip(lines, pts1, pts2):

        color = tuple(np.random.randint(0, 255, 3).tolist())

        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])

        image_colour_1 = cv.line(color1, (x0, y0), (x1, y1), color, 1)
        imag1_with_lines = cv.circle(image_colour_1, (int(pt1[0]), int(pt1[1])), 5, color, -1)
        imag2_with_lines = cv.circle(image_colour_1, (int(pt2[0]), int(pt2[1])), 5, color, -1)
        
    return imag1_with_lines, imag2_with_lines

def un_rectified_epiplines(imag1, imag2, src_pts1, src_pts2):

    lines1 = cv.computeCorrespondEpilines(
        src_pts2.reshape(-1, 1, 2), 2, best_F_matrix)
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = drawlines(imag1, imag2, lines1, src_pts1, src_pts2)

    lines2 = cv.computeCorrespondEpilines(
        src_pts1.reshape(-1, 1, 2), 1, best_F_matrix)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawlines(imag2, imag1, lines2, src_pts2, src_pts1)

    result = np.concatenate((img3, img5), axis=1)

    return result

before_rectified_epiplines = un_rectified_epiplines(gray1, gray2, best_points_calc1, best_points_calc2)

show_image_with_resized_window(before_rectified_epiplines, "before epilines")
wait_key_or_close("before epilines")

cv.imwrite("beforeepilines.png", before_rectified_epiplines)

def disparity_depth(imag1, imag2, baseline, focal_length ):
    window = 5
    disparity_range = 64

    disparity_map = np.zeros_like(imag1)

    for y in range(window//2, imag1.shape[0]-window//2):
        for x in range(window//2, imag1.shape[1]-window//2):
            
            left_image_block = imag1[y-window//2:y+window//2+1, x-window//2:x+window//2+1]
            best_ssd = float('inf')
            best_match_x = -1
        
            for d in range(disparity_range):
                if x-d < window//2:
                    break
                right_image_block = imag2[y-window//2:y+window//2+1, x-window//2-d:x+window//2+1-d]
                ssd_score = np.sum((left_image_block - right_image_block)**2)
                if ssd_score < best_ssd:
                    best_ssd = ssd_score
                    best_match_x = x - d
            disparity_map[y,x] = x - best_match_x

    disparity_map = (disparity_map - np.min(disparity_map)) / (np.max(disparity_map) - np.min(disparity_map)) * 255
    disparity_map = disparity_map.astype(np.uint8)

    epsilon = 1e-3
    depth_map = (baseline * focal_length) / (disparity_map + epsilon)

    depth_map_normalized = cv.normalize(depth_map, None, 0, 255, cv.NORM_MINMAX)

    return disparity_map, depth_map_normalized, depth_map

baseline = 536.62/1000  
focal_length = 1733.74 

disparity_map, depth_map_normalized, depth_map = disparity_depth(img1_rectified, img2_rectified, baseline, focal_length)

cv.imwrite('disparity_grayscale.png', disparity_map)

show_image_with_resized_window(disparity_map, "disparity_grayscale")
wait_key_or_close("disparity_grayscale")

heatmap = cv.applyColorMap(disparity_map, cv.COLORMAP_HOT)

cv.imwrite('disparity_heatmap.png', heatmap)

show_image_with_resized_window(heatmap, "disparity_heatmap")
wait_key_or_close("disparity_heatmap")   


cv.imwrite('depth_grayscale.png', depth_map_normalized)

show_image_with_resized_window(depth_map_normalized, "depth_grayscale")
wait_key_or_close("depth_grayscale")

depth_map_heatmap = cv.applyColorMap(depth_map.astype(np.uint8), cv.COLORMAP_HOT)
cv.imwrite('depth_heatmap.png', depth_map_heatmap)

show_image_with_resized_window(depth_map_heatmap, "depth_heatmap")
wait_key_or_close("depth_heatmap")





     



