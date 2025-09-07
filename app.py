!pip install boto3

#Import and mount the Drive
from google.colab import drive
drive.mount('/content/drive')

#Import sys. 
#Import Open Cv for Sift to Morph
import sys
if 'google.colab' in sys.modules:
    import subprocess
    subprocess.call("pip install -U opencv-python".split())

#To be searched
from random import randrange

#To view the images
import matplotlib.pyplot as plt

#Import mpimg for matplotlyb for image maping
import matplotlib.image as mpimg

#Import Numpy 
import numpy as np

#import os for directory processing
import os

#Import CV2 for image processing
import cv2

#Import Image from PIL for image conversion
from PIL import Image

#To import cv2_imshow to portrait the images
from google.colab.patches import cv2_imshow

#Import boto3 for text extraction
import boto3

#Import pandas for DataFrame creation adn Extarction
import pandas as pd

parent_dir = '/content/drive/MyDrive/ML Training/ScannedDocProject/pdfByName'
# folders to file by form page
filing_dir=os.path.join(parent_dir,'Filed')
Form_A_dir = os.path.join(filing_dir, 'FormA')
Form_B_dir = os.path.join(filing_dir, 'FormB')
Form_C_dir = os.path.join(filing_dir, 'FormC')
Form_D_dir = os.path.join(filing_dir, 'FormD')
Form_E_dir = os.path.join(filing_dir, 'FormE')
Form_C_file_names = os.listdir(Form_C_dir)
Form_C_file_paths = [os.path.join(Form_C_dir, file_name) for file_name in Form_C_file_names[:len(Form_C_file_names)]]

Form_A_file_names=os.listdir(Form_A_dir)
Form_A_file_paths=[os.path.join(Form_A_dir, file_name) for file_name in Form_A_file_names[:len(Form_A_file_names)]]

Form_B_file_names=os.listdir(Form_B_dir)
Form_B_file_paths=[os.path.join(Form_B_dir,file_name) for file_name in Form_B_file_names[:len(Form_B_file_names)]]

Form_D_file_names=os.listdir(Form_D_dir)
Form_D_file_paths=[os.path.join(Form_D_dir,file_name) for file_name in Form_D_file_names[:len(Form_D_file_names)]]

Form_E_file_names=os.listdir(Form_E_dir)
Form_E_file_paths=[os.path.join(Form_E_dir,file_name) for file_name in Form_E_file_names[:len(Form_E_file_names)]]


#Signing into AWS as client IP for textract
client=boto3.client('textract',region_name='us-west-2',aws_access_key_id='My_id',aws_secret_access_key='G/Mykey')

#List File Paths
paths=[Form_A_file_paths,Form_B_file_paths,Form_C_file_paths,Form_D_file_paths,Form_E_file_paths]

#Boxes per form (y1, y2, x1, x2)
# Note: Textract uses a different coordinate system (x, y, width, height) with percentages.
# The original code's cropping boxes are in pixels. For this example, we'll assume the
# original box coordinates were in (top, bottom, left, right) pixel values.
# The new logic finds text within these pixel ranges after processing the whole image.
Boxes_A=[(0,1650,825,2200),(750,1650,1100,2200),(1150,1650,1750,2200),(0,1400,1000,1650),(0,1050,700,1400),(750,1100,1200,1300),(1200,1100,1600,1300)]
Boxes_B=[(1100,375,1300,550),(1200,600,1600,825),(1150,800,1600,1150),(1250,1150,1650,1400)]
Boxes_C=[(325,425,1600,550),(325,550,1600,650),(350,775,1600,850),(490,1100,875,1300)]
Boxes_D=[(200,400,350,700),(1100,800,1225,950),(200,1025,450,1300),(475,1020,800,1300)]
Boxes_E=[(200,580,650,710),(610,575,825,700),(800,575,1010,700),(1300,590,1075,700)]

#List of Boxes
Boxes=[Boxes_A,Boxes_B,Boxes_C,Boxes_D,Boxes_E]

#Labels
Name=[]
MiddleName=[]
LastName=[]
Language=[]
City=[]
State=[]
ZipCode=[]
SeasonalWorker=[]
Income=[]
Funds=[]
Assistance=[]
When=[]
Where=[]
DueDate=[]
Rent=[]
Checks=[]
Shared=[]
ObligCS=[]
PayCS=[]
Type=[]
Amount=[]
StillOwed=[]
Insurance=[]

#List of Labels per form
Labels_A=[Name,MiddleName,LastName,Language,City,State,ZipCode] 
Labels_B=[SeasonalWorker,Income,Funds,Assistance]
Labels_C=[When,Where,DueDate,Rent] 
Labels_D=[Checks,Shared,ObligCS,PayCS]
Labels_E=[Type,Amount,StillOwed,Insurance]

#List of Forms Label
Labels=[Labels_A,Labels_B,Labels_C,Labels_D,Labels_E]

#List of Ids per Form:
ID_A=[]
ID_B=[]
ID_C=[]
ID_D=[]
ID_E=[]

#list of Id Lists
IDs=[ID_A,ID_B,ID_C,ID_D,ID_E]


def get_text_within_box(blocks, box_coords, image_width, image_height):
    """
    Extracts text blocks that fall within a given pixel bounding box.
    
    Args:
        blocks (list): The list of Block objects from the Textract response.
        box_coords (tuple): A tuple of (y1, x1, y2, x2) in pixel coordinates.
        image_width (int): The width of the full image in pixels.
        image_height (int): The height of the full image in pixels.
    
    Returns:
        str: A concatenated string of all text found within the box.
    """
    y1_px, x1_px, y2_px, x2_px = box_coords
    extracted_text = []

    for block in blocks:
        # Check if the block is a LINE of text
        if block['BlockType'] == 'LINE':
            geometry = block['Geometry']['BoundingBox']
            
            # Convert Textract's relative coordinates (0-1) to pixels
            block_left_px = int(geometry['Left'] * image_width)
            block_top_px = int(geometry['Top'] * image_height)
            block_right_px = int((geometry['Left'] + geometry['Width']) * image_width)
            block_bottom_px = int((geometry['Top'] + geometry['Height']) * image_height)

            # Check if the block is within the specified pixel box
            if (block_left_px >= x1_px and block_top_px >= y1_px and
                block_right_px <= x2_px and block_bottom_px <= y2_px):
                extracted_text.append(block['Text'])
    
    return ' '.join(extracted_text) if extracted_text else 'No Response'

def get_ID(image_path, ID_list):
    """
    Extracts the ID from the filename and appends it to the ID list.
    """
    ID_list.append(os.path.basename(image_path).split('p')[0])


def full_extraction(paths, boxes, labels, ids):
    """
    Processes all documents, extracts text, and populates the data lists.
    """
    for form_index in range(len(paths)):
        for file_path in paths[form_index]:
            try:
                # Open the image to get its dimensions
                with open(file_path, 'rb') as raw_image:
                    img_data = raw_image.read()
                    pil_image = Image.open(file_path)
                    image_width, image_height = pil_image.size
                
                # Make a single Textract API call for the whole document
                response = client.detect_document_text(Document={'Bytes': img_data})
                blocks = response['Blocks']

                # Extract text for each defined box on the form
                for box_index in range(len(boxes[form_index])):
                    box_coords = boxes[form_index][box_index]
                    extracted_text = get_text_within_box(
                        blocks, box_coords, image_width, image_height
                    )
                    labels[form_index][box_index].append(extracted_text)
                
                # Get the document ID
                get_ID(file_path, ids[form_index])

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                # Append 'No Response' for each field if an error occurs
                for label_list in labels[form_index]:
                    label_list.append('No Response')
                get_ID(file_path, ids[form_index])

# Run the new, optimized extraction function
full_extraction(paths, Boxes, Labels, IDs)

# --- Data Consolidation and Output ---
# This part of the code remains largely the same
# as the final output structure is identical.

cols = [IDs[0]] + Labels[0] + Labels[1] + Labels[2] + Labels[3] + Labels[4]

col_names=['ID','Name','MiddleName','LastName','Language','City','State','ZipCode', 
'SeasonalWorker','Income','Funds','Assistance',
'When','Where','DueDate','Rent', 
'Checks','Shared','ObligCS','PayCS',
'Type','Amount','StillOwed','Insurance']  

# Check for length consistency before creating DataFrame
if all(len(c) == len(cols[0]) for c in cols):
    df = pd.DataFrame(list(zip(*cols)), columns=col_names)
    df.to_csv('demo.csv', index=False)
    print("Data extracted and saved to demo.csv")
else:
    print("Error: Column lengths are not consistent. Cannot create DataFrame.")
    print("Column lengths:", [len(c) for c in cols])
