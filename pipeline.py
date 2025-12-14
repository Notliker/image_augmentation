import cv2
import numpy as np

from p1_noise_image import (
    GaussianNoiseAugmentation, 
    ImpulseNoiseAugmentation, 
    RayleighNoiseAugmentation, 
    ExponentialNoiseAugmentation
)
from p3_equalization_n_gamma_correction import (
    HistogramEqualizationAugmentation, 
    GammaCorrectionAugmentation
)
from p5_color_converter import (
    RGBtoGrayScaleAugmentation, 
    RGBtoBinaryAugmentation
)
from p7_image_gradients import (
    SobelGradientAugmentation, 
    PrevittGradientAugmentation
)
from p9_geometric_changes import (
    ScalingAugmentation, 
    TranslationAugmentation, 
    RotationAugmentation, 
    GlassEffectAugmentation, 
    Wave1Augmentation, 
    Wave2Augmentation, 
    MotionBlurAugmentation
)

class AugmentationFactory:
    @staticmethod
    def create(name, params):
        #mapping dict
        registry = {
            #1
            "gaussian_noise": GaussianNoiseAugmentation,
            "impulse_noise": ImpulseNoiseAugmentation,
            "rayleigh_noise": RayleighNoiseAugmentation,
            "exponential_noise": ExponentialNoiseAugmentation,
            
            #3
            "histogram_equalization": HistogramEqualizationAugmentation,
            "gamma_correction": GammaCorrectionAugmentation,
            
            #5
            "rgb_to_gray": RGBtoGrayScaleAugmentation,
            "rgb_to_binary": RGBtoBinaryAugmentation,
            
            #7
            "sobel": SobelGradientAugmentation,
            "previtt": PrevittGradientAugmentation,
            
            #9
            "scaling": ScalingAugmentation,
            "translation": TranslationAugmentation,
            "rotation": RotationAugmentation,
            "glass_effect": GlassEffectAugmentation,
            "wave1": Wave1Augmentation,
            "wave2": Wave2Augmentation,
            "motion_blur": MotionBlurAugmentation
        }

        if name not in registry:
            raise ValueError(f"Error! Unknown augmentation type: {name}")

        # augmentation_class - one of augmentation classes we need
        augmentation_class = registry[name]
        try:
            return augmentation_class(**params) #create obj
        except TypeError as e:
            raise ValueError(f"Error! Incorrect params for {name}: {e}")


class AugmentationPipeline:
    def __init__(self):
        self.strategies = []

    def build_from_json(self, config_list):
        """
        [
            {"name": "rotation", "params": {"angle": 90}},
            {"name": "gaussian_noise", "params": {"sigma": 20}}
        ]
        """
        self.strategies = []
        for item in config_list:
            name = item.get("name")
            params = item.get("params", {})
            
            try:
                strategy = AugmentationFactory.create(name, params)
                self.strategies.append(strategy)
            except Exception as e:
                print(f"Error creating '{name}': {e}")

    def run(self, image):
        #start augmentations
        if image is None:
            raise ValueError("Image is None")
            
        current_image = image.copy()
        for strategy in self.strategies:
            print(f"Applying {strategy.__class__.__name__}...")
            current_image = strategy.apply(current_image)
            
        return current_image

if __name__ == "__main__":
    #example
    pipeline_config = [
        {
            "name": "rotation", 
            "params": {"angle": 45}
        },
        {
            "name": "sobel", 
            "params": {"alpha": 0.5}
        },
        {
            "name": "glass_effect",
            "params": {"max_dist": 5}
        }
    ]

    # img = cv2.imread("test.jpg")
    img = np.zeros((300, 300, 3), dtype=np.uint8) + 100 #gray square
    cv2.rectangle(img, (50, 50), (250, 250), (0, 0, 255), -1) #red square

    #augs here
    pipeline = AugmentationPipeline()
    pipeline.build_from_json(pipeline_config)
    result = pipeline.run(img)

    #show results
    cv2.imshow("Original", img)
    cv2.imshow("Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()