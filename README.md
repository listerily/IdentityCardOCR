# Identity Card OCR
This project could locate Identity Card in an image and recognizes ID number as well as name, gender, etc.  
This OCR recognizer could ONLY recognize ID cards which is used in China Mainland.

# How does it work?
1. Locator locates card shaped items in an image by edge detection and transforms that quadrilateral into a rectangle.  
2. Preprocessor processes this rectangle image, reducing noises and removing backgrounds. Note that preprocessor would output a binary image.  
3. Cropper extracts number, name and nationality images from that binary image.  
4. Character Segmentation: Our method would decompose an image of a sequence of characters into subimages of individual characters based on pixel density. Specially, in the segementation of ID Numbers, we used K-Means on horizontal axis for better segmentation results.  
5. CNN Classification: Using our trained models, the program would classify individual images of characters and numbers.  
6. Driver: Driver is used to integrate all above algorithms.  
7. Backend: A flask backend app used to response recognition requests from Android App.  
8. Android App: This app captures image and post it to the backend, after which we could receive results from the backend.  

# Deployment Instructions
+ A linux server with at least 2GiB memory is required.  
+ Clone this repository and train the classifier models. Put all trained models under trained_models/ directory.  
+ Install all python dependencies and start a flask server.  
+ Modify the backend address in App source code into your own server address and then rebuild it.  
