
import cv2
from wave import wavelet
from scipy import signal
import numpy as np
wave = wavelet(4, 0, 21)
def isStressed(image):
    if(len(image)<15):
        return False
    res = signal.fftconvolve(image,wave,'same').real[10:-10,:]
    line=[]
    lineind=[]
    for col in range(len(res[0])):
        cmin=2000
        ind=0
        for row in range(len(res)):
            if(res[row][col]<cmin):
                cmin=res[row][col]
                ind = row            
        #res[ind][col]=-cmin
        line.append(-cmin)
        lineind.append(ind)
    avg = (sum(line)+0.0)/len(line)    
    x_points = []
    y_points = []
    for i in range(len(line)):
        if(line[i]>=avg):
            y_points.append(lineind[i])
            x_points.append(i)
            #res[lineind[i]][i]=100  
    if(np.polyfit(x_points,y_points,2)[0]<=-0.0015):
        return False
    return True

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    
mouth_cascade = cv2.CascadeClassifier('Mouth.xml')
cap = cv2.VideoCapture(0) 
counter=0   
stresspoints=0
while(True):
    status,img = cap.read()
    if(not status):
        break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.1,minSize=(200,200))
    new_gray=img
    mpic=np.array([0])
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray,minSize=(40,40))
        e = 0
        eyeset = []
        for (ex,ey,ew,eh) in eyes:
            if(h-ey<180):#hardcoded, lame, whatever
                continue
            if(e==2):
                break
            e+=1
            eyeset.append([ex,roi_color[ey-15:ey+eh+15, ex-10:ex+ew+10]])
            cv2.rectangle(roi_color,(ex-10,ey-15),(ex+ew+10,ey+eh+15),(0,255,0),2)
#        if(counter%10==0):
#            if(len(eyeset)>1 and eyeset[0][0]>eyeset[1][0]):
#                repic = eyeset[0][1]
#                lepic = eyeset[1][1]
#                cv2.imwrite("stressed/eyes/l_im"+str(counter/10)+".png",lepic)
#                cv2.imwrite("stressed/eyes/r_im"+str(counter/10)+".png",repic)
#            
            
       # print str(eyeline), " is the eyeline"
        mouths = mouth_cascade.detectMultiScale(roi_gray,minNeighbors=5)
        for (ex,ey,ew,eh) in mouths:
            if(h-ey>100):#hardcoded, lame, whatever
                continue
           
            if(counter%10==0):          
#                print counter/10
                mpic=roi_gray[ey-30:ey+eh+30,ex-30:ex+ew+30]
                if(isStressed(mpic)):
                    stresspoints+=1
                    print "+1 Stress Point"
                    if(stresspoints>7):
                        print "Take a break! Go have a walk."
#                cv2.imwrite("stressed/mouths/im"+str(counter/10)+".png",mpic)
#                print "Snap!"
            if(counter%100==0):
                stresspoints=0
            counter+=1
            
            cv2.rectangle(roi_color,(ex-30,ey-30),(ex+ew+30,ey+eh+30),(0,0,255),2)
            
            break
        
        
    
    cv2.imshow('frame',img)
    k = cv2.waitKey(1)
    key = chr(k & 255)
    if key == 'q':
        break
        

    
cv2.destroyAllWindows()

        
        
        
        
        