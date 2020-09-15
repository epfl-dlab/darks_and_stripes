import boto3
import xmltodict
import pandas as pd
import sys

def update_qualification(filename):
    
    workers = pd.read_csv(filename, index_col="workerId")
    
    for index, worker in workers.iterrows():
        print(index)
        response = client.associate_qualification_with_worker(
            QualificationTypeId='3SRUE8JDW6EFRKAEBWA3HB8ASJP5OA',
            #QualificationTypeId='3RVVHO1V7S1LCQ80JEK422CLLU3GWG',	
            WorkerId=index,
            IntegerValue=1,
            SendNotification=False
        )
        '''
        response = client.disassociate_qualification_from_worker(
            WorkerId=index,
            QualificationTypeId='3FVY20EBM5TP1V30YF2SNZX1W2MYGG'
        )
        '''
        

if __name__ == "__main__":
    filename = sys.argv[1]

    client = boto3.client('mturk', region_name='us-east-1')
    update_qualification(filename)