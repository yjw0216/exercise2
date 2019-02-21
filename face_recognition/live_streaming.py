# live_streaming.py

from flask import Flask, render_template, Response
import cv2
import face_recog


app = Flask(__name__,static_url_path='/static')

@app.route('/')
def index():
    num_lst=[0,1,2,3,4,5,6,7,8,9]
    return render_template('index.html',num_lst=num_lst)

def gen(fr):
    while True:
        jpg_bytes = fr.get_jpg_bytes()[0]
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpg_bytes + b'\r\n\r\n')
def gen2(fr_t):
    while True:
        jpg_bytes = fr_t.get_jpg_bytes()[1]
        yield (b'--frame_trim\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpg_bytes + b'\r\n\r\n')

@app.route('/tmp')
@app.route('/video_feed/<int:tmp>')# #int has been used as a filter that only integer will be passed in the url otherwise it will give a 404 error                                 #뒤에 <int::>적어줘야 파라미터 전달이 된다.
def video_feed(tmp=1):
    cap = cv2.VideoCapture('http://192.168.0.73:4747/video') 
    if tmp == 1:
        return Response(gen(face_recog.FaceRecog(cap)),
                mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return Response(gen2(face_recog.FaceRecog(cap)),
                mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/load_img')
def load_img():
    return Response(test(),mimetype='multipart/x-mixed-replace; boundary=frame')
        
'''
@app.route('/trim_frame_feed')
def trim_frame_feed():
    return Response(gen2(face_recog.FaceRecog()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
'''
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
