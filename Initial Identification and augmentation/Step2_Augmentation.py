import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.preprocessing.image import array_to_img
from pathlib import Path
import shutil

train_dir = Path(r'D:\YV\DSGP-IIT\disease and pest detection\All Collected Dataset')
save_dir = train_dir.parent / 'augmented_for_ml'

IMAGE_SIZE = (224, 224)
TARGET_COUNT = 1300

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

save_dir.mkdir(exist_ok=True)

for class_path in train_dir.iterdir():
    if not class_path.is_dir():
        continue

    class_name = class_path.name
    class_save_path = save_dir / class_name
    class_save_path.mkdir(exist_ok=True)

    original_images = [f for f in class_path.glob('*') if f.is_file()]
    original_count = len(original_images)

    print(f"\n=== Class: {class_name} ({original_count} images) ===")

    if original_count == 0:
        print("⚠️ Empty folder — skipping!")
        continue

    # Copy originals
    print("Copying originals...")
    for img_path in original_images:
        try:
            shutil.copy(img_path, class_save_path / img_path.name)
        except:
            pass

    images_needed = TARGET_COUNT - original_count
    if images_needed <= 0:
        print("Already sufficient — skipping augmentation")
        continue

    print(f"Generating {images_needed} new images...")

    generated = 0
    img_index = 0

    while generated < images_needed:
        img_path = original_images[img_index % original_count]
        img = load_img(img_path, target_size=IMAGE_SIZE)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)

        # Generate a batch
        batch = next(datagen.flow(x, batch_size=1))

        # Convert array to image
        new_img = array_to_img(batch[0])

        # Generate custom filename
        filename = class_save_path / f"aug_{generated}.jpg"

        # Save explicitly
        new_img.save(filename)

        generated += 1
        img_index += 1

    print(f"✓ DONE — {class_name} now has {TARGET_COUNT} images")

print("\n🎉 ALL CLASSES PROCESSED — FILES ARE SAVED SUCCESSFULLY ✔")
print(f"Output at: {save_dir}")
