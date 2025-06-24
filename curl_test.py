import requests
#pip install requests

# Replace with the actual path to your image
#image_path = 'your_image.jpg'
image_path = "testimages/tst_img_640_1.jpg"

# Open the image file in binary mode
with open(image_path, 'rb') as img:
    files = {'file': img}
    response = requests.post('http://127.0.0.1:8081/predict', files=files)

# Print the response from the server
print(response.status_code)
print(response.text)
