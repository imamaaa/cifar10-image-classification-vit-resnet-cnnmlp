from flask import Flask, render_template, request, redirect, url_for
import torch
from torchvision import transforms
from PIL import Image
import os
import shutil

app = Flask(__name__, static_url_path='/uploads', static_folder='./uploads')
app.config['UPLOAD_FOLDER'] = './uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global dictionary to store model paths and types
models = {
    "Vision Transformer (ViT)": ("best_vit_checkpoint.pth", "vit"),
    "Custom CNN": ("cnn_model_state_dict.pth", "cnn"),
    "ResNet": ("resnet18_state_dict.pth", "resnet"),
}

transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize to (32, 32) to match training
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Classes for CIFAR-10
classes = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

@app.route("/")
def index():
    """Instructions Page"""
    return render_template("instructions.html")

@app.route("/upload", methods=["GET", "POST"])
def upload():
    """Upload and Model Selection Page"""
    if request.method == "POST":
        model_choice = request.form.get("model")
        files = request.files.getlist("images")

        # Clear uploads folder
        shutil.rmtree(app.config['UPLOAD_FOLDER'])
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

        # Save uploaded files
        for file in files:
            if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Allow PNG, JPEG, JPG
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))

        # Redirect to results page with model choice
        return redirect(url_for("results", model=model_choice))

    return render_template("upload.html", models=models.keys())

@app.route("/results/<model>")
def results(model):
    global models
    model_path, model_type = models.get(model, (None, None))
    if model_path is None:
        return f"Error: Model '{model}' not found!", 404

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # Load the model
        if model_type == "vit":
            from vit_model import VisionTransformer
            loaded_model = VisionTransformer()  # Initialize the ViT model
            try:
                # Load the state dictionary
                state_dict = torch.load(model_path, map_location=device)
                
                # Remove "module." prefix if it exists
                new_state_dict = {}
                for key, value in state_dict.items():
                    new_key = key[len("module."):] if key.startswith("module.") else key
                    new_state_dict[new_key] = value

                # Load the modified state_dict into the model
                loaded_model.load_state_dict(new_state_dict)
            except Exception as e:
                return f"Error loading model '{model}': {e}", 500


        elif model_type == "cnn":
            from custom_cnn import CustomCNN
            loaded_model = CustomCNN()
            state_dict = torch.load(model_path, map_location=device)
            loaded_model.load_state_dict(state_dict)
        elif model_type == "resnet":
            from resnet18_model import CustomResNet18
            loaded_model = CustomResNet18(num_classes=10)
            state_dict = torch.load(model_path, map_location=device)
            loaded_model.load_state_dict(state_dict)

        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    except Exception as e:
        return f"Error loading model '{model}': {e}", 500

    loaded_model.eval().to(device)

    correct_images = []
    incorrect_images = []

    # Process each image
    for image_name in os.listdir(app.config['UPLOAD_FOLDER']):
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_name)

        # Extract ground truth label from the file name
        try:
            ground_truth = int(image_name.split("_label_")[-1].split(".")[0])
        except ValueError:
            return f"Error: Unable to extract label from '{image_name}'", 500

        image = Image.open(image_path).convert("RGB")
        
        # Update input transformation for ViT
        input_transform = transforms.Compose([
            transforms.Resize((224, 224)) if model_type == "vit" else transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        input_tensor = input_transform(image).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            output = loaded_model(input_tensor)
            pred = output.argmax(dim=1).item()

        # Classify as correct or incorrect
        if ground_truth == pred:
            correct_images.append((os.path.relpath(image_path, './uploads'), classes[pred]))
        else:
            incorrect_images.append((os.path.relpath(image_path, './uploads'), f"Predicted: {classes[pred]} | Actual: {classes[ground_truth]}"))

    return render_template(
        "results.html", correct=correct_images, incorrect=incorrect_images
    )


if __name__ == "__main__":
    app.run(debug=True)
