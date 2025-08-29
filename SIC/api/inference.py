# SIC/api/inference.py

import tensorflow as tf
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from pathlib import Path
import torch
import numpy as np

# ==================== 1. DEFINE SCENE ANALYSIS CLASS ====================
class SceneAnalysisModel:
    def __init__(self, model_path: Path):
        """
        Initializes the pipeline with:
        1. A scene classification model (ResNet50 transfer learning).
        2. A BLIP captioning model (HuggingFace).
        """
        # --- Load Scene Classification Model ---
        print(f"Loading scene classification model from: {model_path}")
        self.classifier = tf.keras.models.load_model(model_path)
        self.class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
        print("Scene classification model loaded successfully ✅")

        # --- Load BLIP Image Captioning Model ---
        print("Loading BLIP Image Captioning model (may take time on first run)...")
        self.caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.caption_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        print("BLIP model loaded successfully ✅")

        # --- Use GPU if available ---
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.caption_model.to(self.device)

    # --------- Scene Classification ---------
    def classify_scene(self, pil_image: Image.Image) -> str:
        """
        Classifies the given image into one of the scene categories.
        """
        rgb_image = pil_image.convert("RGB").resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(rgb_image)
        img_array = tf.expand_dims(img_array, axis=0)  # batch dimension
        img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

        preds = self.classifier(img_array, training=False)  # normal prediction
        class_idx = tf.argmax(preds, axis=1).numpy()[0]
        return self.class_names[class_idx]

    # --------- MC Dropout Prediction ---------
    def predict_with_uncertainty(self, pil_image: Image.Image, n_iter: int = 30):
        """
        Performs multiple predictions with Dropout active (MC Dropout)
        and returns class, confidence, and uncertainty.
        """
        rgb_image = pil_image.convert("RGB").resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(rgb_image)
        img_array = tf.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

        predictions = []
        for _ in range(n_iter):
            pred = self.classifier(img_array, training=True)  # Dropout active
            predictions.append(pred.numpy())

        predictions = np.vstack(predictions)
        mean_preds = np.mean(predictions, axis=0)
        predicted_idx = np.argmax(mean_preds)
        predicted_class = self.class_names[predicted_idx]
        confidence = mean_preds[predicted_idx]
        uncertainty = np.std(predictions[:, predicted_idx])

        return predicted_class, float(confidence), float(uncertainty)

    # --------- Caption Generation ---------
    def generate_description(self, pil_image: Image.Image) -> str:
        """
        Generates a descriptive caption for the given image using BLIP.
        """
        rgb_image = pil_image.convert("RGB")
        inputs = self.caption_processor(rgb_image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = self.caption_model.generate(**inputs, max_new_tokens=50)

        caption = self.caption_processor.decode(output[0], skip_special_tokens=True)
        return caption

    # --------- Full Vision-Text Pipeline ---------
    def analyze_scene(self, pil_image: Image.Image, mc_dropout: bool = False) -> dict:
        """
        Returns both the classified scene category and a natural language description.
        Set mc_dropout=True to get confidence and uncertainty estimates.
        """
        if mc_dropout:
            scene_label, confidence, uncertainty = self.predict_with_uncertainty(pil_image)
        else:
            scene_label = self.classify_scene(pil_image)
            confidence, uncertainty = None, None

        caption = self.generate_description(pil_image)

        return {
            "scene_category": scene_label,
            "confidence": confidence,
            "uncertainty": uncertainty,
            "caption": caption
        }

# ==================== 2. USAGE EXAMPLE ====================
if __name__ == "__main__":
    model_weights_path = Path("SIC/models/weights/transfer_learning_resnet50.h5")
    example_image_path = Path(__file__).parent.parent / "data/seg_pred/seg_pred/1649.jpg"

    if not example_image_path.exists():
        print(f"❌ Error: Example image not found at {example_image_path}")
    else:
        # --- Initialize Pipeline ---
        model = SceneAnalysisModel(model_path=model_weights_path)

        # --- Run Analysis with MC Dropout ---
        image = Image.open(example_image_path)
        results = model.analyze_scene(image, mc_dropout=True)

        print("\n====== Scene Analysis Results ======")
        print(f"🖼️  Image: {example_image_path.name}")
        print(f"🏞️  Predicted Scene Category: {results['scene_category']}")
        if results['confidence'] is not None:
            print(f"💡 Confidence Score: {results['confidence']:.4f}")
            print(f"🤔 Uncertainty Estimate: {results['uncertainty']:.4f}")
        print(f"📝 Generated Caption: {results['caption']}")
