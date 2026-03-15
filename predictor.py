import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess


# ===============================
# Load Trained Models
# ===============================
mobilenet_model = load_model(r'C:\Users\YV\PycharmProjects\DSGP_GROUP_36\mobilenet.h5')
efficientnet_model = load_model(r'C:\Users\YV\PycharmProjects\DSGP_GROUP_36\efficientnet_best.h5')
resnet50_model = load_model(r'C:\Users\YV\PycharmProjects\DSGP_GROUP_36\resnet50_best.h5')


# ===============================
# Class Labels
# ===============================
classes = [
    'Anthracnose', 'Gray Blight', 'Green mirid bug', 'Heliopeltis', 'Red Rust',
    'Red Spider', 'Sunlight Scorching', 'Tea leaf blight', 'Tea red leaf spot',
    'Tea red scab', 'Thrips', 'algal leaf', 'bird eye spot', 'brown blight',
    'gray light', 'white spot'
]


# ===============================
# Image Preprocessing Helper
# ===============================
def load_image_for_model(img_path, model, preprocess_func):
    _, height, width, _ = model.input_shape
    img = image.load_img(img_path, target_size=(height, width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_func(img_array)
    return img_array


# ===============================
# Prediction Function
# ===============================
def predict_disease(
    img_path,
    weight_mobilenet=0.3,
    weight_efficientnet=0.4,
    weight_resnet50=0.3,
    temperature=2.0
):

    # MobileNet
    img_m = load_image_for_model(img_path, mobilenet_model, mobilenet_preprocess)
    pred_m = mobilenet_model.predict(img_m)[0]

    pred_m_scaled = np.exp(np.log(pred_m + 1e-8) / temperature)
    pred_m_scaled /= np.sum(pred_m_scaled)

    idx_m = np.argmax(pred_m)


    # EfficientNet
    img_e = load_image_for_model(img_path, efficientnet_model, efficientnet_preprocess)
    pred_e = efficientnet_model.predict(img_e)[0]

    pred_e_scaled = np.exp(np.log(pred_e + 1e-8) / temperature)
    pred_e_scaled /= np.sum(pred_e_scaled)

    idx_e = np.argmax(pred_e)


    # ResNet50
    img_r = load_image_for_model(img_path, resnet50_model, resnet50_preprocess)
    pred_r = resnet50_model.predict(img_r)[0]

    pred_r_scaled = np.exp(np.log(pred_r + 1e-8) / temperature)
    pred_r_scaled /= np.sum(pred_r_scaled)

    idx_r = np.argmax(pred_r)


    # Weighted Soft Voting
    combined_pred = (
        weight_mobilenet * pred_m_scaled +
        weight_efficientnet * pred_e_scaled +
        weight_resnet50 * pred_r_scaled
    )

    final_index_soft = np.argmax(combined_pred)
    final_class_soft = classes[final_index_soft]
    final_conf_soft = combined_pred[final_index_soft]


    # Hard Voting
    votes = [idx_m, idx_e, idx_r]
    final_index_hard = max(set(votes), key=votes.count)
    vote_count = votes.count(final_index_hard)

    final_class_hard = classes[final_index_hard]
    final_conf_hard = vote_count / len(votes)


    # Final Decision
    if vote_count >= 2:
        final_class = final_class_hard
        final_confidence = final_conf_hard
    else:
        final_class = final_class_soft
        final_confidence = final_conf_soft

    return final_class, float(final_confidence)