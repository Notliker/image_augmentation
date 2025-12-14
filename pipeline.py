import cv2
import numpy as np
import os

from utils.p1_noise_image import (
    GaussianNoiseAugmentation, 
    ImpulseNoiseAugmentation, 
    RayleighNoiseAugmentation, 
    ExponentialNoiseAugmentation
)
from utils.p2_noise_clearing import (
    AverageBlurAugmentation, 
    GaussianBlurAugmentation, 
    MedianBlurAugmentation
)
from utils.p3_equalization_n_gamma_correction import (
    HistogramEqualizationAugmentation, 
    GammaCorrectionAugmentation
)
from utils.p4_YIQ_processong import YIQAugmentationWrapper
from utils.p5_color_converter import (
    RGBtoGrayScaleAugmentation, 
    RGBtoBinaryAugmentation
)
from utils.p6_colorize import ColorRestorationAugmentation
from utils.p7_image_gradients import (
    SobelGradientAugmentation, 
    PrevittGradientAugmentation
)
from utils.p8_image_mixing import ImageMixingAugmentation
from utils.p9_geometric_changes import (
    ScalingAugmentation, 
    TranslationAugmentation, 
    RotationAugmentation, 
    GlassEffectAugmentation, 
    Wave1Augmentation, 
    Wave2Augmentation, 
    MotionBlurAugmentation
)
from utils.p10_custom import AutoMixingAugmentation

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
            
            #2
            "average_blur": AverageBlurAugmentation,
            "gaussian_blur": GaussianBlurAugmentation,
            "median_blur": MedianBlurAugmentation,
            
            #3
            "histogram_equalization": HistogramEqualizationAugmentation,
            "gamma_correction": GammaCorrectionAugmentation,
            
            #5
            "rgb_to_gray": RGBtoGrayScaleAugmentation,
            "rgb_to_binary": RGBtoBinaryAugmentation,
            
            #6
            "color_restoration": ColorRestorationAugmentation,
            
            #7
            "sobel": SobelGradientAugmentation,
            "previtt": PrevittGradientAugmentation,
            
            #8
            "image_mixing": ImageMixingAugmentation,
            
            #9
            "scaling": ScalingAugmentation,
            "translation": TranslationAugmentation,
            "rotation": RotationAugmentation,
            "glass_effect": GlassEffectAugmentation,
            "wave1": Wave1Augmentation,
            "wave2": Wave2Augmentation,
            "motion_blur": MotionBlurAugmentation,
            
            #10
            "auto_mixing": AutoMixingAugmentation
        }
        #4
        if name == "yiq_wrapper":
            inner_name = params.pop("inner_name")
            inner_params = params.pop("inner_params", {})
            apply_to = params.pop("apply_to", "Y")
            
            #recursive creation of inner augmentation (that been wrapped)!
            inner_aug_obj = AugmentationFactory.create(inner_name, inner_params)
            
            return YIQAugmentationWrapper(inner_aug_obj, apply_to)

        #main creation
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
            params = item.get("params", {}).copy()
            
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

# if __name__ == "__main__":
#     #example
#     pipeline_config = [
#         {
#             "name": "rotation", 
#             "params": {"angle": 45}
#         },
#         {
#             "name": "sobel", 
#             "params": {"alpha": 0.5}
#         },
#         {
#             "name": "glass_effect",
#             "params": {"max_dist": 5}
#         }
#     ]

#     # img = cv2.imread("test.jpg")
#     img = np.zeros((300, 300, 3), dtype=np.uint8) + 100 #gray square
#     cv2.rectangle(img, (50, 50), (250, 250), (0, 0, 255), -1) #red square

#     #augs here
#     pipeline = AugmentationPipeline()
#     pipeline.build_from_json(pipeline_config)
#     result = pipeline.run(img)

#     #show results
#     cv2.imshow("Original", img)
#     cv2.imshow("Result", result)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     #gradient img
#     img = np.zeros((300, 300, 3), dtype=np.uint8)
#     for y in range(300):
#         for x in range(300):
#             img[y, x] = [x * 255 // 300, 0, y * 255 // 300] # B G R

#     #green rectangl
#     cv2.rectangle(img, (100, 100), (200, 200), (0, 255, 0), -1)

#     #JSON config
#     # noise only to Y (sigma=50)
#     pipeline_config = [
#         {
#             "name": "yiq_wrapper",
#             "params": {
#                 "apply_to": "Y",
#                 "inner_name": "gaussian_noise",  #augmentation method
#                 "inner_params": {
#                     "sigma": 50
#                 }
#             }
#         }
#     ]

#     #start
#     pipeline = AugmentationPipeline()
#     pipeline.build_from_json(pipeline_config)
    
#     print("Запускаем пайплайн...")
#     result_img = pipeline.run(img)

#     #show results
#     cv2.imshow("Original", img)
#     cv2.imshow("YIQ Luma Noise (Grayscale noise on Color)", result_img)
    
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

def get_all_images(dataset_root):
    """finds all images in folder recursive"""
    images = []
    for root, _, files in os.walk(dataset_root):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                images.append(os.path.join(root, file))
    return images

def get_valid_reference_path(dataset_root):
    """any ref image from dir"""
    imgs = get_all_images(dataset_root)
    return imgs[0] if imgs else None

def get_valid_dataset_path(dataset_root):
    """first valid dir for 8_ImageMixing"""
    for root, _, files in os.walk(dataset_root):
        if any(f.lower().endswith(('.jpg', '.png')) for f in files):
            return root
    return dataset_root


if __name__ == "__main__":
    dataset_path = "dataset"
    output_path = "output"
    os.makedirs(output_path, exist_ok=True)

    images_to_process = get_all_images(dataset_path)
    if not images_to_process:
        print(f"Error: No images found in '{dataset_path}'")
        exit()

    ref_file = get_valid_reference_path(dataset_path)
    ref_folder = get_valid_dataset_path(dataset_path)

    if not ref_file:
        print("Error: Dataset is empty, cannot perform mixing/color restoration tests.")
        exit()

    print(f"Found {len(images_to_process)} images.")
    print(f"Using reference file: {ref_file}")
    print(f"Using reference folder: {ref_folder}")

    config_noise = [
        {"name": "gaussian_noise", "params": {"sigma": 15}},
        {"name": "impulse_noise", "params": {"noise_percentage": 0.05}},
        {"name": "gaussian_blur", "params": {"kernel_size": 5}}
    ]

    config_geo = [
        {"name": "rotation", "params": {"angle": 45}},
        {"name": "scaling", "params": {"scale_x": 0.8, "scale_y": 0.8}},
        {"name": "glass_effect", "params": {"max_dist": 5}}
    ]

    config_yiq = [
        {
            "name": "yiq_wrapper",
            "params": {
                "apply_to": "Y",
                "inner_name": "wave1",
                "inner_params": {"amplitude": 10, "period": 20}
            }
        }
    ]

    config_mix = [
        {
            "name": "image_mixing", 
            "params": {
                "mode": "chess", 
                "dataset_path": ref_folder,
                "patch_size": [50, 50]
            }
        },
        {
            "name": "auto_mixing",
            "params": {
                "dataset_path": ref_folder,
                "win_size": [100, 100],
                "alpha": 0.2
            }
        }
    ]

    config_color = [
        {"name": "rgb_to_gray", "params": {}},
        {"name": "sobel", "params": {"alpha": 1.0}},
        {"name": "color_restoration", "params": {"reference_path": ref_file}}
    ]

    all_configs = [
        ("noise_blur", config_noise),
        ("geometry", config_geo),
        ("yiq_wave", config_yiq),
        ("mixing", config_mix),
        ("gradients", config_color)
    ]

    pipeline = AugmentationPipeline()

    #cycle
    for img_path in images_to_process:
        print(f"\nProcessing: {img_path}")
        original = cv2.imread(img_path)
        
        if original is None:
            print(f"Failed to load {img_path}")
            continue

        filename = os.path.basename(img_path)
        name, ext = os.path.splitext(filename)

        for config_name, config_data in all_configs:
            print(f"  -> Pipeline: {config_name}")
            
            #rebuild pipeline for new config
            pipeline.build_from_json(config_data)
            
            try:
                result = pipeline.run(original)
                
                #save
                save_name = f"{name}_{config_name}{ext}"
                save_path = os.path.join(output_path, save_name)
                cv2.imwrite(save_path, result)
            except Exception as e:
                print(f"CRITICAL ERROR in {config_name}: {e}")

    print("\nDone. Check 'output' folder for reslts.")