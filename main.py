import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# Define a dictionary for animal descriptions. The key is the classification label from the model.
ANIMAL_DESCRIPTIONS = {
    'golden_retriever': "A friendly and intelligent breed of dog, known for its beautiful golden coat and gentle temperament. Golden retrievers are often used as guide dogs and are popular family pets.",
    'tiger': "A large, carnivorous cat recognized by its distinctive dark vertical stripes on orange-brown fur. The tiger is an apex predator, native to parts of Asia. It's a powerful and solitary hunter.",
    'zebra': "A member of the horse family distinguished by its unique black-and-white striped coat. Zebras are native to Africa and are known for their social nature, living in herds.",
    'elephant': "The largest living land animal, elephants are known for their long trunks, large ears, and tusks. They are highly intelligent and social creatures, forming strong family bonds within their herds.",
    'dog': "A domesticated mammal with a wide variety of breeds. Dogs are known for their loyalty, companionship, and keen sense of smell. They are often called 'man's best friend'.",
    'cat': "A small, carnivorous mammal known for its agility and grace. Cats are popular pets worldwide and come in many different breeds, sizes, and colors. They are both solitary hunters and social animals.",
    'bear': "A large, powerful mammal with a thick coat of fur. Bears are found in various habitats and are known for their omnivorous diet. Different species include black bears, brown bears, and polar bears.",
    'lion': "A large, powerful cat native to Africa and India. The male is easily recognizable by its magnificent mane. Lions are the only cats that live in social groups called prides.",
    'wolf': "A large canid native to the wilderness of North America and Eurasia. Wolves are apex predators that live and hunt in packs. They are known for their distinctive howling calls.",
    'hippopotamus': "A large, semi-aquatic mammal native to sub-Saharan Africa. Hippos are known for their huge jaws and teeth. Despite their bulky appearance, they are very fast on land and in water."
}

def get_animal_description(label):
    """
    Returns a short description for a given animal label.
    Args:
        label (str): The predicted class label from the model.
    Returns:
        str: A description of the animal, or a default message if not found.
    """
    # The label comes in the format 'nXXXXXXXX_label', we only need the label part.
    if '_' in label:
        label = label.split('_', 1)[1]

    # Check if the label is in our predefined dictionary.
    if label in ANIMAL_DESCRIPTIONS:
        return ANIMAL_DESCRIPTIONS[label]
    else:
        # Fallback to a general message for labels not in our dictionary.
        return f"A photo of a '{label}'. This animal is a fascinating creature, but I don't have a specific description for it in my database."


def classify_image(image_path):
    """
    Loads an image from a given path, preprocesses it, and makes a prediction.
    Args:
        image_path (str): The file path to the image to classify.
    """
    if not os.path.exists(image_path):
        print(f"Error: The file '{image_path}' was not found.")
        return

    # Load the image and resize it to the target size for the model
    try:
        img = image.load_img(image_path, target_size=(224, 224))
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Convert the image to a numpy array
    img_array = image.img_to_array(img)
    # The model expects a batch of images, so we expand the dimensions
    img_array = np.expand_dims(img_array, axis=0)

    # Preprocess the image for the MobileNetV2 model
    processed_image = preprocess_input(img_array)

    # Load the pre-trained MobileNetV2 model
    print("Loading the MobileNetV2 model...")
    model = MobileNetV2(weights='imagenet')

    # Make predictions
    print("Classifying the image...")
    predictions = model.predict(processed_image)

    # Decode the top 3 predictions
    decoded_predictions = decode_predictions(predictions, top=3)[0]

    # Get the top prediction
    top_prediction = decoded_predictions[0]
    predicted_label = top_prediction[1]
    prediction_confidence = top_prediction[2] * 100

    print(f"\nPrediction: {predicted_label.replace('_', ' ').capitalize()} (Confidence: {prediction_confidence:.2f}%)")

    # Get and print the description
    description = get_animal_description(predicted_label)
    print("\n--- Animal Description ---")
    print(description)
    print("--------------------------")

    # Print top 3 predictions for more context
    print("\nTop 3 predictions:")
    for i, (imagenet_id, label, confidence) in enumerate(decoded_predictions):
        print(f"{i+1}. {label.replace('_', ' ').capitalize()}: {confidence:.2f}%")


if __name__ == "__main__":
    # Prompt the user for the image file path
    print("Please make sure your image file is in the same directory as this script.")
    image_file = input("Enter the name of the image file (e.g., photo.jpg): ")
    classify_image(image_file)
