import cv2
import pickle
from flask import Flask, render_template, Response, request
from threading import Thread

# initialise our app
app = Flask(__name__)
print("Getting Video")
camera = cv2.VideoCapture(0)

resolution = [480, 270]

do_recognition = False

#face_box_data = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
face_box_data = [[0, 0, 0, 0]]
face_box_color = (0,255,0)
text_box_color = (255, 255, 255)
eye_box_color = (255, 0, 0)
stroke_width = 2

text_data = [[0, 0]]
font = cv2.FONT_HERSHEY_COMPLEX

eye_box_data = [[0, 0, 0, 0]]

#x_resolution = 1280
#y_resolution = 720

def generate_frames():

    # works best with front facing
    face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

    # to get the region of interest of our faces for our eyes
    eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')

    # bring in our recogniser
    recogniser = cv2.face.LBPHFaceRecognizer_create()
    recogniser.read("trainer.yml")

    labels = {}

    # bring in our label_ids from the training
    with open("labels.pickle", 'rb') as file: # rb is to read#
        labels = pickle.load(file)
        # currently, this dictionary is in the format {name : id}
        # we want to reverse this to be {id : name}
        labels = {v:k for k,v in labels.items()}

    name = labels[0]
    print(labels)

    while True:
        # capture the video
        ret,frame = camera.read()

        # change resolution of frame
        frame = cv2.resize(frame, (resolution), fx=0, fy=0, interpolation = cv2.INTER_CUBIC)
        #frame = cv2.resize(frame, (500, 500), fx=0, fy=0, interpolation = cv2.INTER_CUBIC)
        #print(resolution)
        #global frame

        #frame = camera_read()

        global do_recognition
        do_recognition = not do_recognition

        if do_recognition == True:

            # put frame into greyscale
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            faces = face_cascade.detectMultiScale(grey, scaleFactor=1.5, minNeighbors=5)

            #face_number = 0

            # iterate through faces
            # x,y,w,h, are values of where a face is
            for(x, y, w, h) in faces:

                #print(x, y, w, h)
                region_of_interest_grey = grey[y:y+h, x:x+w] # y:y+h, x:x+h is the square of the region
                region_of_interest_colour = frame[y:y+h, x:x+w]

                # we predict who is the face
                # we get the label back, and the confidence of the prediction
                id_, confidence = recogniser.predict(region_of_interest_grey)
                if confidence >= 50: # and confidence <= 85:
                    #print(id_)
                    #print(labels[id_])

                    # put the name of the person on the frame
                    #font = cv2.FONT_HERSHEY_COMPLEX
                    name = labels[id_]
                    #colour = (255, 255, 255)
                    #stroke_width = 2
                    #text_data[0] = x
                    #text_data[1] = y
                    new_text = [0, 0]
                    new_text[0] = x
                    new_text[1] = y
                    text_data.append(new_text)
                    cv2.putText(frame, name, (x,y), font, 1, face_box_color, stroke_width, cv2.LINE_AA)

                #img_item_grey = "image.png"
                #img_item_colour = "image_colour.png"
                #cv2.imwrite(img_item_grey, region_of_interest_grey)
                #cv2.imwrite(img_item_colour, region_of_interest_colour)

                # set up for drawing around a face
                #colour = (0,255,0) #BGR not RGB
                #stroke_width = 2

                width = x + w
                height = y + h

                new_face = [0, 0, 0, 0]
                new_face[0] = x
                new_face[1] = y
                new_face[2] = width
                new_face[3] = height

                face_box_data.append(new_face)

                # drawing a rectangle on the frame
                cv2.rectangle(frame, (x, y), (width, height), face_box_color, stroke_width) # x and y are the starting coordinates, width and height are the ending coordinates

                #face_number += 1

                #face_box_data.pop()

                # get the eyes
                eyes = eye_cascade.detectMultiScale(region_of_interest_grey)

                for (ex,ey,ew,eh) in eyes:
                    new_eye = [0, 0, 0, 0]
                    new_eye[0] = ex
                    new_eye[1] = ey
                    new_eye[2] = ex + ew
                    new_eye[3] = ey + eh
                    eye_box_data.append(new_eye)
                    cv2.rectangle(region_of_interest_colour, (ex, ey), (ex+ew, ey+eh), eye_box_color, stroke_width)
        
        text_number = len(text_data)
        if text_number > 2:
            text_data.pop(0)
        
        for text in text_data:
            cv2.putText(frame, name, (text[0], text[1]), font, 1, face_box_color, stroke_width, cv2.LINE_AA)

        face_number = len(face_box_data)
        if face_number > 2:
            face_box_data.pop(0)

        for face in face_box_data:
            # draw the last known box
            cv2.rectangle(frame, (face[0], face[1]), (face[2], face[3]), face_box_color, stroke_width)
        
        eye_number = len(eye_box_data)
        if eye_number > 4:
            eye_box_data.pop(0)

        # display the video
        #cv2.imshow("Frame", frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame=buffer.tobytes()
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        #print("Displaying")
        #image_display(frame)


        #k = cv2.waitKey(1)
        #if k == ord('q'):
        #    break

# do camera reading
def camera_read():
    # capture the video
    ret,frame = camera.read()

    # change resolution of frame
    frame = cv2.resize(frame, (resolution), fx=0, fy=0, interpolation = cv2.INTER_CUBIC)

    return frame

# display the frame
def image_display(frame):
    print("Displaying")
    ret, buffer = cv2.imencode('.jpg', frame)
    frame=buffer.tobytes()
    yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

class NewThread(Thread):
    # constructor
    def __init__(self):
        # execute the base constructor
        Thread.__init__(self)
        # set a default value
        self.value = None
 
    # function executed in a new thread
    def run(self):
        # block for a moment
        #sleep(1)
        # store data in an instance variable
        self.value = 'Hello from a new thread'

# whenever we get to the url specified, the function below will run
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods =["GET", "POST"])
def changeOptions():
    if request.method == "POST":
       resolution_option = request.form.get("example")
       resolution_numbers = resolution_option.split()
       print(resolution_numbers)
       resolution[0] = int(resolution_numbers[0])
       resolution[1] = int(resolution_numbers[1])
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run()