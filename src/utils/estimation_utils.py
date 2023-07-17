import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from config import cfg

def calculate_shift_error(image1, image2):
    gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)

    # Вычислим оптический поток
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)

    shift_error_x = np.mean(np.abs(flow[..., 0]))  # Вычислим ошибку сдвига как среднее абсолютное значение оптического потока по соответствующим осям
    shift_error_y = np.mean(np.abs(flow[..., 1]))

    return shift_error_x, shift_error_y

@torch.no_grad()
def calculate_and_plot_shift_errors(model, transform, data):
    errors_x = []
    errors_y = []
    model.eval()

    for img_src_path, img_dst_path in data:
        image = Image.open(img_src_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(cfg["device"])

        transformed_image_tensor = model(image_tensor)

        image_display = image_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
        transformed_image_display = transformed_image_tensor.squeeze().cpu().permute(1, 2, 0).numpy()

        image_shift = Image.open(img_dst_path).convert('RGB')
        image_shift_tensor = transform(image_shift).unsqueeze(0).to(cfg["device"])
        image_shift_display = image_shift_tensor.squeeze().cpu().permute(1, 2, 0).numpy()

        shift_error_x, shift_error_y = calculate_shift_error(image_shift_display, transformed_image_display)
        errors_x.append(shift_error_x * 2048 / cfg['image_size'])
        errors_y.append(shift_error_y * 2048 / cfg['image_size'])

    # Построение гистограммы ошибок сдвига
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(errors_x, bins=50, color='blue', alpha=0.7)
    plt.title('Histogram of Shift X Errors')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)

    plt.subplot(1, 2, 2)
    plt.hist(errors_y, bins=50, color='red', alpha=0.7)
    plt.title('Histogram of Shift Y Errors')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()
