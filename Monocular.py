import matplotlib.pyplot as plt
from PIL import Image
import torch
from transformers import GLPNImageProcessor, GLPNForDepthEstimation

def load_model(model_name="vinvino02/glpn-nyu"):
    feature_extractor = GLPNImageProcessor.from_pretrained(model_name)
    model = GLPNForDepthEstimation.from_pretrained(model_name)
    return feature_extractor, model

def preprocess_image(image_path, max_height=480):
    image = Image.open(image_path)
    new_height = min(image.height, max_height)
    new_height -= new_height % 32
    new_width = int(new_height * image.width / image.height)
    diff = new_width % 32
    new_width = new_width - diff if diff < 16 else new_width + 32 - diff
    return image.resize((new_width, new_height))

def predict_depth(image, feature_extractor, model, pad=16):
    inputs = feature_extractor(images=image, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth.squeeze().cpu().numpy() * 1000.0
    return predicted_depth[pad:-pad, pad:-pad], image.crop((pad, pad, image.width - pad, image.height - pad))

def display_results(image, depth_map):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image)
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    ax[1].imshow(depth_map, cmap='plasma')
    ax[1].set_title('Depth Map')
    ax[1].axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image_path = "S:\\Image reconstruction\\3dcheck.jpeg"
    feature_extractor, model = load_model()
    image = preprocess_image(image_path)
    depth_map, cropped_image = predict_depth(image, feature_extractor, model)