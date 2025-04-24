import os
import json
import boto3
import base64
import numpy as np
import asyncio
from facenet_pytorch import MTCNN
from PIL import Image
import logging

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize SQS client
sqs = boto3.client('sqs')

class face_detection:
    def __init__(self):
        # Initialize MTCNN model for face detection
        self.mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20)

    async def face_detection_func(self, test_image_path, output_path):
        """
        Detects face in the given image and saves the detected face.
        Args:
            test_image_path (str): Path to the input image.
            output_path (str): Directory where detected face image will be saved.
        Returns:
            str: Path to the saved face image, or None if no face is detected.
        """
        # Load and preprocess the image
        img = Image.open(test_image_path).convert("RGB")
        img = np.array(img)
        img = Image.fromarray(img)
        
        key = os.path.splitext(os.path.basename(test_image_path))[0].split(".")[0]
        
        # Perform face detection
        face, prob = self.mtcnn(img, return_prob=True, save_path=None)
        
        if face is not None:
            # Ensure the output directory exists
            os.makedirs(output_path, exist_ok=True)
            
            # Normalize and prepare the detected face image
            face_img = face - face.min()
            face_img = face_img / face_img.max()
            face_img = (face_img * 255).byte().permute(1, 2, 0).numpy()
            face_pil = Image.fromarray(face_img, mode="RGB")
            
            # Save the detected face image
            face_img_path = os.path.join(output_path, f"{key}_face.jpg")
            face_pil.save(face_img_path)
            
            return face_img_path
        else:
            return None

# Instantiate face detection object
fd = face_detection()

async def async_handler(event):
    """
    Handles the asynchronous event for face detection.
    Args:
        event (dict): Lambda event input.
    Returns:
        dict: API Gateway compatible response.
    """
    logger.info(f"Received event: {event}")
    body = json.loads(event['body'])
    
    content = body['content']
    request_id = body['request_id']
    filename = os.path.basename(body['filename'])
    
    # Save the incoming base64 image to a temporary file
    temp_image_path = f"/tmp/{filename}"
    with open(temp_image_path, 'wb') as f:
        f.write(base64.b64decode(content))
    
    temp_output_path = "/tmp/detected_faces"
    os.makedirs(temp_output_path, exist_ok=True)
    
    # Run face detection
    face_path = await fd.face_detection_func(temp_image_path, temp_output_path)
    
    if face_path:
        # Encode detected face image to base64
        with open(face_path, 'rb') as f:
            face_data = f.read()
        face_base64 = base64.b64encode(face_data).decode('utf-8')
        
        # Prepare SQS message
        message = {
            'request_id': request_id,
            'face_image': face_base64,
            'filename': filename
        }
        
        # Send message to SQS queue
        response = sqs.send_message(
            QueueUrl=QUEUE_URL,
            MessageBody=json.dumps(message)
        )
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Face detected and sent for recognition',
                'request_id': request_id
            })
        }
    else:
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'No face detected in the image',
                'request_id': request_id
            })
        }

def handler(event, context):
    """
    AWS Lambda synchronous handler.
    Args:
        event (dict): Lambda event input.
        context (object): Lambda context object.
    Returns:
        dict: API Gateway compatible response.
    """
    logger.info("Lambda handler invoked.")
    try:
        # Execute the async handler inside the synchronous Lambda
        result = asyncio.run(async_handler(event))
        return result
    except Exception as e:
        logger.error(f"Error in face detection: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'message': f'Error in face detection: {str(e)}'})
        }
