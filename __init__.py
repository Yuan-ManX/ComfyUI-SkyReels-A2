from .nodes import LoadA2Model, ReferenceImages, A2VideoGenerator, CombineImages, SaveVideo

NODE_CLASS_MAPPINGS = {
    "Load A2 Model": LoadA2Model,
    "Reference Images": ReferenceImages,
    "A2 Video Generator": A2VideoGenerator,
    "Combine Images": CombineImages,
    "Save Video": SaveVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadA2Model": "Load A2 Model",
    "ReferenceImages": "Reference Images",
    "A2VideoGenerator": "A2 Video Generator",
    "CombineImages": Combine Images,
    "SaveVideo": Save Video,
} 

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
