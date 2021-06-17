# creating data
import cv2
import numpy as np


face_classifier = cv2.CascadeClassifier('haar.xml')


def face_extractor(img):
    
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    if faces is ():
        return None
    

    for (x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face


cap = cv2.VideoCapture(0)
count = 0


while True:

    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count += 1
        face = cv2.resize(face_extractor(frame), (200, 200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        
        file_name_path = './faces/' + str(count) + '.jpg'
        cv2.imwrite(file_name_path, face)

        
        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Face Cropper', face)
        
    else:
        print("Face not found")
        pass

    if cv2.waitKey(1) == 13 or count == 100: 
        break
        
cap.release()
cv2.destroyAllWindows()      
print("Collecting Samples Complete")




# training model to recognize your face
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import joblib

data_path = './faces/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

Training_Data, Labels = [], []


for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)


Labels = np.asarray(Labels, dtype=np.int32)



model  = cv2.face_LBPHFaceRecognizer.create()
 
model.train(np.asarray(Training_Data), np.asarray(Labels))
print("Model trained sucessefully")






#creating functionns for the actions
# for sending mail
def mail_sender(pwd):
    from email.message import EmailMessage
    Sender_Email = "email id "
    Reciever_Email = "email id"
    Password = pwd
    newMessage = EmailMessage()                         
    newMessage['Subject'] = "Check out the new logo" 
    newMessage['From'] = Sender_Email                   
    newMessage['To'] = Reciever_Email                   
    newMessage.set_content('someone is trying to enter your computer') 
    with open('yo.jpg', 'rb') as f:
        image_data = f.read()
        image_type = imghdr.what(f.name)
        image_name = f.name
    newMessage.add_attachment(image_data, maintype='image', subtype=image_type, filename=image_name)
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
    
        smtp.login(Sender_Email, Password)              
        smtp.send_message(newMessage)
     
     
# sending watsapp message
import pywhatkit as kit
def send_watsapp(x,y):
    kit.sendwhatmsg("+91**********", "someone is in front of your cam!",x,y)
    
    
    
#creating instance and ebs and attaching
def ins():
    ec2 = boto3.resource('ec2')
    instance = ec2.create_instances(
        ImageId='ami-0ad704c126371a549',
        MinCount=1,
        MaxCount=1,
        InstanceType='t2.micro',
        SecurityGroupIds=['sg-026dd3773ee684723'],
        SubnetId='subnet-c8c5cca0',
    )
    return(instance[0].id)
    
 def ebs():
    ebs=ec2.create_volume(
        AvailabilityZone='ap-south-1a',
        Size=4,
        VolumeType='standard')
    return(ebs.id)
    
def attach(vid,inid):
    volume =ec2.Volume(vid)
    attach_ebs=volume.attach_to_instance(
        Device='/dev/sdh',
        InstanceId= inid,
        VolumeId=vid)
    return (done)
    
def final():
    inid=ins()
    vid=ebs()
    time.sleep(200)
    attach(vid,inid)    
    
    
    
    
    
# running final model predecting ,getting accuracy,and running action
import cv2
import numpy as np
import os
import smtplib
import imghdr
import pywhatkit as kit
import boto3
import imghdr


pwd=input('Enter your email account password: ')
face_classifier = cv2.CascadeClassifier('haar.xml')

def face_detector(img, size=0.5):
    
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return img, []
    
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200, 200))
    return img, roi



cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    
    image, face = face_detector(frame)
    
    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        results = vimal_model.predict(face)
        
        
        if results[1] <= 500:
            confidence = int( 100 * (1 - (results[1]/400)) )
            display_string = str(confidence) + '% Confident it is user'
             
            
        cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (255,120,150), 2)
        
        if confidence > 80:
            cv2.putText(image, "Hey shubhyansh", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.imshow('Face Recognition', image )
            cv2.imwrite('filename.jpg',imge)
            mail_sender(pw)
            send_watsapp()
            break
            
            
            
            
         
        else:
            
            cv2.putText(image, "I dont know, who r u", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            cv2.imshow('Face Recognition', image )
            final()

    except:
        cv2.putText(image, "No Face Found", (220, 120) , cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.putText(image, "looking for face", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.imshow('Face Recognition', image )
        pass
        
    if cv2.waitKey(55) == 13: 
        break
        
print('successfully created')        
cap.release()
cv2.destroyAllWindows()     



