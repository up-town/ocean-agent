import json
import boto3
import os
import uuid
import random

sqs_client = boto3.client('sqs')
sqsFifoUrl = os.environ.get('sqsFifoUrl')
print('sqsFifoUrl: ', sqsFifoUrl)

nqueue = os.environ.get('nqueue')

def lambda_handler(event, context):
    print('event: ', json.dumps(event))

    for record in event['Records']:
        eventId = str(uuid.uuid1())
        print('eventId: ', eventId)
                
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']
        print('bucket: ', bucket)
        print('key: ', key)
                
        s3EventInfo = {
            'event_id': eventId,
            'event_timestamp': record['eventTime'],
            'bucket': bucket,
            'key': key,
            'type': record['eventName']
        }
        
        # push to SQS
        try:
            print('sqsFifoUrl: ', sqsFifoUrl)            
            
            sqs_client.send_message(  # fifo
                QueueUrl=sqsFifoUrl, 
                MessageAttributes={},
                MessageDeduplicationId=eventId,
                MessageGroupId="s3event",
                MessageBody=json.dumps(s3EventInfo)
            )
            print('Successfully push the queue message: ', json.dumps(s3EventInfo))
            
        except Exception as e:        
            print('Fail to push the queue message: ', e)
        
    return {
        'statusCode': 200
    }