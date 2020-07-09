import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import math as m
##########################################Variables################################################
pi = m.pi
##########################################Functions################################################
def ycord(x,rho,theta):
    #y = (rho - x.cos(theta))/sin(theta)
    #theta is in degrees. It must be converted to radians by: radians = degrees*(pi/180)
    #theta = theta*(pi/180)
    return (rho - x*np.cos(theta))/np.sin(theta)
def rad2deg(rad):
    return rad*(180/m.pi)
def getTranslationMatrix(tx,ty):
    return np.float32([[1,0,tx],[0,1,ty]])
##########################################Main Code################################################

cap = cv.VideoCapture('onboard2.avi')

#the order of the read frame. It is used to create the title of the frame to save
imgcount = 0
#Read the frame and store it in Ic. If a problem in reading this frame occur, then the next while loop won't run
ret, Ic = cap.read()
count1 = 0
#Compute image width W and image height H
W, H = (Ic.shape[1],Ic.shape[0])
#Compute center coordinates
xc, yc = W/2, H/2
#Create a Video Writer Object
fourcc = cv.VideoWriter_fourcc('M','J','P','G')
vid_HL = cv.VideoWriter('onboard2_with_HL.avi',fourcc,30.0,(W,H),True)
vid_stab = cv.VideoWriter('onboard2_stab.avi',fourcc,30.0,(Ic.shape[1],Ic.shape[0]),True)

while ret == True:
    Ic_no_overlay = Ic #Create a copy of the original color image Ic. It's useful as Ic would be overlayed by a line
    count1 = count1 + 1
    Ig = cv.cvtColor(Ic, cv.COLOR_BGR2GRAY) #discuss this function
    Is1 = cv.medianBlur(Ig,5*1)
    Is2 = cv.medianBlur(Ig,5*3)
    Is3 = cv.medianBlur(Ig,5*5)
    E1 = cv.Canny(Is1,70,120)
    E2 = cv.Canny(Is2,70,120)
    E3 = cv.Canny(Is3,70,120)
    Ew = (np.float16(E1) + np.float16(E2) + np.float16(E3))/3
    Ew = np.uint8(Ew)
    ret, Eth = cv.threshold(Ew,170,255,0)
    #Find lines coordinates
    #IMPORTANT:cv.HoughLines returns rho and theta coordinates of all found lines in the form of a 3D array
    #We name this 3D array lines
    #The number of rows in lines is equal to the number of lines. This is the 1st dimension
    #The number of columns is equal to two. This is the 2nd dimension
    #The 3rd dimension has a size of 1.
    lines = cv.HoughLines(Eth,1,m.pi/180,30)

    if lines is not None:
        #linedim is a tuple of 3 elements (#_of_rows,size_of_the_3rd_dim,#of_columns_inside_each_row)
        linesdim = lines.shape
        #The goal the reshape command is to reshape the array lines from 3D to 2D. Because:
            #There is no reason to keep it 3D
            #Reshaping it in 2D makes it easy for unpacking the line coordinates rho and theta
        lines = np.reshape(lines,(linesdim[0],linesdim[2]))
        #print(lines)
        xs, xe = (0,Ig.shape[1])
        #Take rho and theta of the first element in lines.
        rho, theta = lines[0]
        ys = int(ycord(xs,rho,theta))
        ye = int(ycord(xe,rho,theta))
        #Draw a the found line on the original image Ic
        #cv.line(Ic,(xs,ys),(xe,ye),(0,0,255),5)
    imtitle = 'Result' + str(imgcount) + '.jpg'
    imgcount = imgcount + 1
    #vid_HL.write(Ic)
    # Stabilization process:
        #Compute in degrees D_alpha = alpha_ref - alpha_det; alpha_ref = 0
    if ye < ys:
        sign = 1
    else:
        sign = -1
    S = m.sqrt((W**2) + ((ye-ys)**2))
    alpha_ref = 0
    alpha_det = m.acos(W/S) * sign
    D_alpha = rad2deg(alpha_ref - alpha_det) # This is the value by which to rotate the image
        #Compute in pixels D_Y = Y_ref - Y_det; Y_ref = Image_Height/2
            #The Y_det is characterized by (x,y) coordinates.
            #Y_det is always measured from the middle from the top middle of the image to the buttom
            #Because the measure starts from the middle, this means that the corresponding x = ImageWidth/2
    Y_ref = yc
    Y_det = ycord(W/2,rho,theta)
    D_Y = Y_ref - Y_det
        #Stabilize the image using the angular "D_alpha" and positional error "D_Y"
            #Get the rotational Matrix Rm
    Rm = cv.getRotationMatrix2D((xc,yc),D_alpha,1)
            #Get the translational matrix Tx
    Tx = getTranslationMatrix(0,D_Y)
        #Crate a stabilized Image (We stabilized the no overlayed original image Ic_no_overlay)
    I_stab = cv.warpAffine(Ic_no_overlay,Rm,(W,H))
    I_stab = cv.warpAffine(I_stab,Tx,(W,H))
        #Save the stabilized Video
    vid_stab.write(I_stab)
#Read the next frame (if it exists)
    ret, Ic = cap.read()
        #print(rho,theta)
    #rho = x.cos(theta) + y.sin(theta)
    #Steps:
        #Get end points (xs,ys) and (xe,ye) of a line with a known rho and theta
            #calculation of (xs,ys)
                #xs = 0
                #ys = (rho - xs.cos(theta))/sin(theta)
            #calculation of (xe,ye)
                #xe = number of the image columns (assuming that x indexes colums)
                #ye = (rho - xe.cos(theta))/sin(theta)
cap.release()
vid_HL.release()
print("End of the code")
