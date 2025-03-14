import os
import sys
import cv2
import numpy as np
import pygame
import tensorflow as tf

# Константы
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43

# Словарь соответствия классов и названий знаков
SIGN_LABELS = {
    0: "Ограничение скорости 20 км/ч / Speed Limit 20 km/h",
    1: "Ограничение скорости 30 км/ч / Speed Limit 30 km/h",
    2: "Ограничение скорости 50 км/ч / Speed Limit 50 km/h",
    3: "Ограничение скорости 60 км/ч / Speed Limit 60 km/h",
    4: "Ограничение скорости 70 км/ч / Speed Limit 70 km/h",
    5: "Ограничение скорости 80 км/ч / Speed Limit 80 km/h",
    6: "Конец ограничения скорости 80 км/ч / End of Speed Limit 80 km/h",
    7: "Ограничение скорости 100 км/ч / Speed Limit 100 km/h",
    8: "Ограничение скорости 120 км/ч / Speed Limit 120 km/h",
    9: "Обгон запрещен / No passing",
    10: "Обгон грузовикам запрещен / No passing for trucks",
    11: "Главная дорога / Right-of-way at the next intersection",
    12: "Уступи дорогу / Yield",
    13: "Стоп / Stop",
    14: "Проезд запрещен / No vehicles",
    15: "Запрещено движение грузовиков / No trucks",
    16: "Ограничение въезда / No entry",
    17: "Объезд справа / General caution",
    18: "Опасный поворот налево / Dangerous curve to the left",
    19: "Опасный поворот направо / Dangerous curve to the right",
    20: "Двойной поворот / Double curve",
    21: "Горный перевал / Bumpy road",
    22: "Скользкая дорога / Slippery road",
    23: "Сужение дороги / Road narrows on the right",
    24: "Дорожные работы / Road work",
    25: "Светофорное регулирование / Traffic signals",
    26: "Пешеходный переход / Pedestrians",
    27: "Дети / Children crossing",
    28: "Пересечение с велосипедной дорожкой / Bicycles crossing",
    29: "Осторожно, снег / Beware of ice/snow",
    30: "Дикие животные / Wild animals crossing",
    31: "Ограничение скорости отменено / End speed + passing limits",
    32: "Обязательное движение направо / Turn right ahead",
    33: "Обязательное движение налево / Turn left ahead",
    34: "Движение только прямо / Ahead only",
    35: "Движение прямо или направо / Go straight or right",
    36: "Движение прямо или налево / Go straight or left",
    37: "Объезд справа / Keep right",
    38: "Объезд слева / Keep left",
    39: "Круговое движение / Roundabout mandatory",
    40: "Ограничение въезда грузовиков / End of no passing",
    41: "Конец запрета на обгон грузовиков / End of no passing by trucks",
    42: "Осторожно, нерегулируемый переход / Attention, unregulated pedestrian crossing"
}

# Проверяем аргументы командной строки
if len(sys.argv) != 3:
    print("Использование: python3 recognize.py model.h5 путь_к_изображению.jpg")
    sys.exit(1)

# Загружаем модель
model_path = sys.argv[1]
image_path = sys.argv[2]

if not os.path.exists(model_path):
    print(f"Error: can't find model file {model_path}.")
    sys.exit(1)

if not os.path.exists(image_path):
    print(f"Error: can't find image {image_path}.")
    sys.exit(1)

print("Loading model...")
model = tf.keras.models.load_model(model_path)

# Загружаем изображение
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

# Подготавливаем для предсказания
img_array = np.array(img, dtype=np.float32) / 255.0
img_array = img_array.reshape(1, IMG_WIDTH, IMG_HEIGHT, 3)

# Делаем предсказание
pred = model.predict(img_array)
predicted_class = np.argmax(pred)

# Получаем название знака
sign_name = SIGN_LABELS.get(predicted_class, "Unknown sign")

# Инициализируем Pygame
pygame.init()
size = (600, 400)
screen = pygame.display.set_mode(size)
pygame.display.set_caption("Traffic Sign Recognition")

# Отображение изображения и результата
image_surface = pygame.image.load(image_path)
image_surface = pygame.transform.scale(image_surface, (300, 300))

# Настройки шрифта
font = pygame.font.Font(None, 30)
text_surface = font.render(sign_name, True, (255, 255, 255))

# Основной цикл Pygame
running = True
while running:
    screen.fill((0, 0, 0))  # Черный фон
    screen.blit(image_surface, (150, 30))  # Размещаем изображение в центре
    screen.blit(text_surface, (20, 350))  # Размещаем текст под изображением

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.display.flip()

pygame.quit()
