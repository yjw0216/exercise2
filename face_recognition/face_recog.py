# face_recog.py
from multiprocessing import Process,Pool
import face_recognition
import cv2
import camera
import os
import numpy as np
import urllib.request

class FaceRecog():
    def __init__(self,FaceRecog):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.camera = camera.VideoCamera()

        self.known_face_encodings = []
        self.known_face_names = []
        self.cap = FaceRecog
        # Load sample pictures and learn how to recognize it.
        dirname = 'knowns'
        files = os.listdir(dirname)
        for filename in files:
            name, ext = os.path.splitext(filename)
            if ext == '.jpg':
                self.known_face_names.append(name)
                pathname = os.path.join(dirname, filename)
                img = face_recognition.load_image_file(pathname)
                face_encoding = face_recognition.face_encodings(img)[0]
                self.known_face_encodings.append(face_encoding)

        # Initialize some variables
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.face_distance = []
        self.process_this_frame = True
        self.i=0
    def __del__(self):
        #del self.camera
        pass
   
    def get_frame(self):
        # Grab a single frame of video
        #frame = self.camera.get_frame()
        #http://10.27.18.4:8080/shot.jpg
        
        ret, frame = self.cap.read()
        
        frame_trim = frame[0:100,0:100]
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        
        
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if self.process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            self.face_locations = face_recognition.face_locations(rgb_small_frame)
            
            #d
            self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)
            
            
            #outCap3 = self.face_encodings[0,0]
            #outCap2= np.array(bytearray(outCap3.read()),dtype=np.uint8)
            #outCap = cv2.imdecode(outCap2,-1)
            #cv2.imwrite('messigray.jpg',outCap)
            self.face_names = []
            for face_encoding in self.face_encodings:
                # See if the face is a match for the known face(s)
                distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                min_value = min(distances)

                # tolerance: How much distance between faces to consider it a match. Lower is more strict.
                # 0.6 is typical best performance.
                name = "Unknown"
                if min_value < 0.6:
                    index = np.argmin(distances)
                    name = self.known_face_names[index]
                self.face_names.append(name)
                self.face_distance.append(min_value)
        self.process_this_frame = not self.process_this_frame

        # Display the results
        for (top, right, bottom, left), name, distance in zip(self.face_locations, self.face_names, self.face_distance):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            #cv2.putText(frame, int(min_value), (left + 12, bottom - 6), font, 1.0, (255, 255, 255), 1)
            print(self.face_locations,'------')
            top=self.face_locations[0][0]*4
            left=self.face_locations[0][3]*4
            right=self.face_locations[0][1]*4
            bottom=self.face_locations[0][2]*4
            frame_trim = frame[top:bottom,left:right]
            # top부터 바텀까지, left부터 right까지 = 빨간 네모 크기
            #cv2.imwrite('11.jpg',frame)
            #cv2.imwrite('%s%s.jpg'%(name,self.i),frame_trim)
            self.i= self.i+1

        return (frame,frame_trim)

    def get_jpg_bytes(self):
        frame = self.get_frame()[0]
        frame_trim = self.get_frame()[1]
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpg = cv2.imencode('.jpg', frame)
        ret2, jpg2 = cv2.imencode('.jpg', frame_trim)
        return (jpg.tobytes(),jpg2.tobytes())
def Test():
        cap = cv2.VideoCapture('http://192.168.0.73:4747/video') 
        face_recog = FaceRecog(cap)
        
        while True:
            frame = face_recog.get_frame()[0]
            frame2 = face_recog.get_frame()[1]
            # show the frame
            cv2.imshow("Frame", frame)
            cv2.imshow("Frame2", frame2)
            #cv2.imshow("Frame", frame2)
            key = cv2.waitKey(1) & 0xFF
            
            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break
           

        # do a bit of cleanup
        cv2.destroyAllWindows()
        #cv2.destroyAllWindows()
        print('finish')
if __name__ == '__main__':
    #face_recog = FaceRecog()
    #face_recog2 = FaceRecog2()
    #print(face_recog.known_face_names)
    Test()
