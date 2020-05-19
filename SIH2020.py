import cv2
import numpy as np
from skimage.feature import (match_descriptors, corner_harris,
                             corner_peaks, ORB, plot_matches)#Contains Algorithm to find descriptors and keypoints
#Another Algo is SIFT
def Evaluate(original,image_to_compare):
    # 1) Check if 2 images are equals
    if original.shape == image_to_compare.shape:
        print("The images have same size and channels")
        difference = cv2.subtract(original, image_to_compare)
        b, g, r = cv2.split(difference)

        if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
            print("The images are completely Equal")
        else:
            print("The images are NOT equal")
    else:
        print("Shape of images aren't equal.")
    # 2) Check for similarities between the 2 images
    #sift = cv2.xfeatures2d.SIFT_create()
    # Initiate ORB detector
    orb = cv2.ORB_create()
    # find the keypoints with ORB
    kp_1 = orb.detect(original,None)

    # compute the descriptors with ORB
    kp_1, desc_1 = orb.compute(original, kp_1)
    orb1 = cv2.ORB_create()
    kp_2 = orb1.detect(image_to_compare,None)
    # compute the descriptors with ORB
    kp_2, desc_2 = orb1.compute(image_to_compare, kp_2)
    flann=cv2.BFMatcher()
    matches = flann.knnMatch(desc_1, desc_2, k=2)

    good_points = []
    for m, n in matches:
        if m.distance < 0.97*n.distance:
            good_points.append(m)

    # Define how similar they are
    number_keypoints = 0
    if len(kp_1) <= len(kp_2):
        number_keypoints = len(kp_1)
    else:
        number_keypoints = len(kp_2)
    T=(kp_1,kp_2,good_points,number_keypoints)
    result = cv2.drawMatches(original, kp_1, image_to_compare, kp_2, good_points, None)


    cv2.imshow("result", cv2.resize(result, None, fx=0.4, fy=0.4))
    cv2.imwrite("feature_matching.jpg", result)


    cv2.imshow("Original", cv2.resize(original, None, fx=0.4, fy=0.4))
    cv2.imshow("Duplicate", cv2.resize(image_to_compare, None, fx=0.4, fy=0.4))
    cv2.waitKey(3000)
    cv2.destroyAllWindows()
    return T

#Input original image
Orimage=input("Enter the expected Image of the Project(Original/.jpg,.jpeg,.png): ")
original = cv2.imread(Orimage)

#image_to_compare = cv2.imread("C:/Users/hp/Desktop/T2.jpg")
#Picture of location before start of construction
print("\t\t\t\tProgress Checking Device of Housing Construction\n\t\t\t\t\t\t\tSikkim" ) 
project_duration=int(input("Enter the total project duration in months: "))#To to complete project
n=int(input("Enter the number of checkpoints: "))#Checkpoints to verify progress
Tcheck_intervals=float(project_duration/n)
addr=input("Enter the address of initial picture of location before the commencement of project(.jpg,.jpeg,.png): ")
image_to_compare=cv2.imread(addr)
Beforeconst=Evaluate(original,image_to_compare)
BCkp_1,BCkp_2,BCgood_points,BCnumber_keypoints=Beforeconst
BCpercentagesimilar=(len(BCgood_points) / BCnumber_keypoints) * 100
#print(BCpercentagesimilar)


#Progress Report of Percentage Completion of work
for i in range(1,n+1):
    print("Picture of Location after",round(Tcheck_intervals*i),"months")
    check1=input("Enter the address of picture of location(.jpg,.jpeg,.png): ")
    image_to_compare=cv2.imread(check1)
    T=Evaluate(original,image_to_compare)
    kp_1,kp_2,good_points,number_keypoints=T
    print("Completion of work after",round(Tcheck_intervals*i),"months: ", ((len(good_points) / number_keypoints) * 100),"%")
    progress=((len(good_points) / number_keypoints) * 100)
if progress>=95: 
    print("Congratulations!! The project has been succesfully completed.")
elif progress>=75:
    print("Project is about to be completed.")
else:
    print("ATTENTION: Incompletion of project within due time period.")
        








#kp_1,kp_2,good_points,number_keypoints=T
#print("Keypoints 1ST Image: " + str(len(kp_1)))
#print("Keypoints 2ND Image: " + str(len(kp_2)))
#print("Good KPs:", len(good_points))
#print("Percent Match: ", (len(good_points) / number_keypoints) * 100)




