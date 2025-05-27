from .nodes import LoadA2Model, ReferenceImages, A2Prompt, NegativePrompt, A2VideoGenerator, CombineImages, SaveVideo

NODE_CLASS_MAPPINGS = {
    "LoadA2Model": LoadA2Model,
    "ReferenceImages": ReferenceImages,
    "A2Prompt": A2Prompt,
    "NegativePrompt": NegativePrompt,
    "A2VideoGenerator": A2VideoGenerator,
    "CombineImages": CombineImages,
    "SaveVideo": SaveVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadA2Model": "Load A2 Model",
    "ReferenceImages": "Reference Images",
    "A2Prompt": "A2 Prompt",
    "NegativePrompt": "Negative Prompt",
    "A2VideoGenerator": "A2 Video Generator",
    "CombineImages": "Combine Images",
    "SaveVideo": "Save Video",
} 

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
