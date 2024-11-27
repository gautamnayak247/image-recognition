from transformers import pipeline

def get_image_details(image_path: str):
    img_recog_pipeline = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    result = img_recog_pipeline(image_path) 
    return result[0]['generated_text']


if __name__ =="__main__":
    print(f'cat: {get_image_details("./image/cat.jpeg")}')
    print(f'dog: {get_image_details("./image/img2.jpeg")}')