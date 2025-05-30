# inference.py (updated to use config)
import torch
from PIL import Image
from torchvision import transforms
from lit_module import AttrClassifier
from config import get_config
import numpy as np
import argparse


class AttributePredictor:
    """Inference class for CelebA attribute prediction."""
    
    def __init__(self, checkpoint_path, config=None, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.cfg = config or get_config()
        
        # Load model
        self.model = AttrClassifier.load_from_checkpoint(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Define transform based on config
        self.transform = transforms.Compose([
            transforms.Resize(self.cfg.model.img_size),
            transforms.CenterCrop(self.cfg.model.img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Attribute names
        self.attr_names = [
            '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
            'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
            'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
            'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
            'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
            'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
            'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
            'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
            'Wearing_Necklace', 'Wearing_Necktie', 'Young'
        ]
    
    def predict(self, image_path, threshold=0.5, top_k=10):
        """
        Predict attributes for a single image.
        
        Args:
            image_path: Path to the image
            threshold: Confidence threshold for binary predictions
            top_k: Number of top predictions to show
            
        Returns:
            dict with predictions and probabilities
        """
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            logits = self.model(img_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        
        # Create results dictionary
        results = {
            'all_probabilities': {attr: float(prob) for attr, prob in zip(self.attr_names, probs)},
            'binary_predictions': {attr: bool(prob > threshold) for attr, prob in zip(self.attr_names, probs)},
            'top_k': []
        }
        
        # Get top-k predictions
        top_indices = np.argsort(probs)[::-1][:top_k]
        for idx in top_indices:
            results['top_k'].append({
                'attribute': self.attr_names[idx],
                'probability': float(probs[idx])
            })
        
        return results
    
    def print_predictions(self, image_path, threshold=0.5):
        """Print formatted predictions."""
        results = self.predict(image_path, threshold=threshold)
        
        print(f"\nPredictions for: {image_path}")
        print("="*50)
        
        print("\nTop 10 Attributes:")
        for i, pred in enumerate(results['top_k'], 1):
            print(f"{i:2d}. {pred['attribute']:25s} {pred['probability']:.3f}")
        
        print(f"\nPositive Attributes (>{threshold}):")
        positive_attrs = [attr for attr, pred in results['binary_predictions'].items() if pred]
        if positive_attrs:
            for attr in positive_attrs:
                print(f"  - {attr:25s} {results['all_probabilities'][attr]:.3f}")
        else:
            print("  No attributes above threshold")


def main():
    parser = argparse.ArgumentParser(description='Predict CelebA attributes for an image')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('--checkpoint', type=str, 
                        default='checkpoints/attribute_classifier/celebahq/last.ckpt',
                        help='Path to model checkpoint')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Confidence threshold for binary predictions')
    args = parser.parse_args()
    
    # Load config
    cfg = get_config()
    
    # Create predictor
    predictor = AttributePredictor(args.checkpoint, config=cfg)
    
    # Make predictions
    predictor.print_predictions(args.image_path, threshold=args.threshold)


if __name__ == '__main__':
    main()
