Depth Estimation using GLPN Transformer

This project demonstrates depth estimation from a single image using the GLPN (Global-Local Path Network) transformer model from Hugging Face.

**Project Structure**

Python Script: Contains the code to load the GLPN model, preprocess the image, perform depth estimation, and display the results.

**Requirements**

Python 3.x

torch (PyTorch)

transformers (Hugging Face Transformers)

Pillow (for image processing)

matplotlib (for displaying images and depth maps)

**Install dependencies using:**

pip install torch transformers Pillow matplotlib

**How to Run**

Set the path to your image in the image_path variable.

**Run the Python script:**

python monocular.py

**Explanation of Code**

load_model(): Loads the GLPN model and feature extractor.

preprocess_image(): Resizes the image while maintaining dimensions divisible by 32.

predict_depth(): Generates a depth map using the loaded model.

display_results(): Displays the original image and the depth map side by side.

**Output**

The output will show the original image alongside its depth map colored using the 'plasma' colormap.

**References**

Hugging Face Transformers

GLPN Model Documentation


