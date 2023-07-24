import torch
import click
import os
import torchvision.transforms as transforms
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PIL import Image
from src.main_pipeline import ShiftNetModule


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@click.command()
@click.option('--img_path', prompt='Image Path', help='Path to the source image file.')
def transform_and_save_image(img_path):
    model = ShiftNetModule()
    model.load_state_dict(torch.load('Data/model_weights/best_model_weights .pth'))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    image = Image.open(img_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        transformed_image_tensor = model(image_tensor)

    transformed_image_display = transformed_image_tensor.squeeze().cpu().permute(1, 2, 0).numpy()

    output_path = os.path.join(os.path.dirname(img_path), "shifted_" + os.path.basename(img_path))
    transformed_image = Image.fromarray((transformed_image_display * 255).astype('uint8'))
    transformed_image.save(output_path)
    print(f"Transformed image saved at {output_path}")


if __name__ == "__main__":
    transform_and_save_image()



