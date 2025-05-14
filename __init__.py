from .nodes import LoadA2Model, ReferenceImages, Prompt, NegativePrompt, A2VideoGenerator, CombineImages, SaveVideo

NODE_CLASS_MAPPINGS = {
    "LoadA2Model": LoadA2Model,
    "ReferenceImages": ReferenceImages,
    "Prompt": Prompt,
    "NegativePrompt": NegativePrompt,
    "A2VideoGenerator": A2VideoGenerator,
    "CombineImages": CombineImages,
    "SaveVideo": SaveVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadA2Model": "Load A2 Model",
    "ReferenceImages": "Reference Images",
    "Prompt": "Prompt",
    "NegativePrompt": "Negative Prompt",
    "A2VideoGenerator": "A2 Video Generator",
    "CombineImages": "Combine Images",
    "SaveVideo": "Save Video",
} 

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
