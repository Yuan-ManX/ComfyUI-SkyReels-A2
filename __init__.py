from .nodes import A2ModelLoader, A2VideoGenerator, SaveMP4Video

NODE_CLASS_MAPPINGS = {
    "A2 Model Loader": A2ModelLoader,
    "A2 Video Generator": A2VideoGenerator,
    "Save MP4 Video": SaveMP4Video,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "A2ModelLoader": "A2 Model Loader",
    "A2VideoGenerator": "A2 Video Generator",
    "SaveMP4Video": "Save MP4 Video",
} 

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
