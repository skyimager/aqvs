
import time
from flask import request,jsonify,Flask

from src.prediction.defect_det_service import DefectDetectionService
from src.utils import utils

dds = DefectDetectionService()

app = Flask(__name__)

@app.route('/')
def index():
    return "Welcome to Defect Detection Api"

@app.route('/detect',methods=['POST'])
def recognise():

    data = request.get_json()
    encoded_content = data['encodedContent']

    start_time= time.time()
    image_arr = utils.get_image(encoded_content)

    respCode = None
    results = None
    final_img = None
    
    try:
        print ("Starting Detection Service!!!")
        results = dds.predict_label(image_arr)
        print ("Detection Service Finished!!!")
        
        if len(results)>0:
            final_img = dds.get_annotated_img()

    except Exception as e:
        print(str(e))
        respCode = {"code":415, "description": "Unable to detect defect."}


    api_response = {'result':results,
                    'annotation':final_img,
                    'respCode': respCode}
    
    print (api_response)
    print ("total API time: ", time.time()-start_time)
    return jsonify(api_response)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=9790)
