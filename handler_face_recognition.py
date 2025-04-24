import os
import json
import boto3
import base64
import torch
import numpy as np
from PIL import Image
import asyncio
import logging

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize SQS client
sqs = boto3.client('sqs')

class face_recognition:
    async def face_recognition_func(self, model_path, model_wt_path, face_img_path):
        """
        Recognizes a face by comparing it against saved embeddings.
        Args:
            model_path (str): Path to the TorchScript face recognition model.
            model_wt_path (str): Path to the saved embeddings and names.
            face_img_path (str): Path to the face image to be recognized.
        Returns:
            str: Recognized person's name, or None if no match found.
        """
        # Load and preprocess the input face image
        face_pil = Image.open(face_img_path).convert("RGB")
        key = os.path.splitext(os.path.basename(face_img_path))[0].split(".")[0]
        
        face_numpy = np.array(face_pil, dtype=np.float32) / 255.0
        face_numpy = np.transpose(face_numpy, (2, 0, 1))  # Convert to (C, H, W) format
        face_tensor = torch.tensor(face_numpy, dtype=torch.float32)
        
        # Load pre-trained face embeddings and names
        saved_data = torch.load(model_wt_path)
        
        # Load the trained TorchScript model
        self.resnet = torch.jit.load(model_path)

        if face_tensor is not None:
            # Generate the face embedding for the input face
            emb = self.resnet(face_tensor.unsqueeze(0)).detach()

            embedding_list = saved_data[0]  # List of known face embeddings
            name_list = saved_data[1]       # Corresponding list of names
            
            # Compute distance between input embedding and all saved embeddings
            dist_list = [torch.dist(emb, emb_db).item() for emb_db in embedding_list]
            
            # Identify the closest match
            min_index = dist_list.index(min(dist_list))
            recognized_name = name_list[min_index]
            
            return recognized_name
        else:
            # No face tensor was created
            return None

# Instantiate the face recognition object
fr = face_recognition()

async def async_handler(event):
    """
    Asynchronous handler for processing face recognition from SQS events.
    Args:
        event (dict): The SQS event input.
    Returns:
        dict: API Gateway-compatible response.
    """
    logger.info("Face recognition Lambda invoked")
    
    # Paths for model and weights inside Lambda environment
    model_path = os.path.join(os.environ.get('LAMBDA_TASK_ROOT', ''), 'resnetV1.pt')
    model_wt_path = os.path.join(os.environ.get('LAMBDA_TASK_ROOT', ''), 'resnetV1_video_weights.pt')

    for record in event['Records']:
        message_body = json.loads(record['body'])
        request_id = message_body['request_id']
        face_base64 = message_body['face_image']
        filename = message_body['filename']

        # Save the incoming face image to a temporary path
        temp_face_path = '/tmp/face.jpg'
        with open(temp_face_path, 'wb') as f:
            f.write(base64.b64decode(face_base64))
        
        # Perform face recognition
        recognized_name = await fr.face_recognition_func(model_path, model_wt_path, temp_face_path)

        # Prepare the result message
        result_message = {
            'request_id': request_id,
            'result': recognized_name if recognized_name else "Unknown"
        }

        # Send the result back via SQS
        response = sqs.send_message(
            QueueUrl=QUEUE_URL,
            MessageBody=json.dumps(result_message)
        )

    return {
        'statusCode': 200,
        'body': json.dumps({'message': 'Face recognition completed successfully'})
    }

def handler(event, context):
    """
    AWS Lambda synchronous handler for face recognition.
    Args:
        event (dict): Lambda event input.
        context (object): Lambda context object.
    Returns:
        dict: API Gateway-compatible response.
    """
    logger.info("Lambda handler invoked for face recognition")
    try:
        # Execute the asynchronous handler
        result = asyncio.run(async_handler(event))
        return result
    except Exception as e:
        logger.error(f"Error in face recognition: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'message': f'Error in face recognition: {str(e)}'})
        }
