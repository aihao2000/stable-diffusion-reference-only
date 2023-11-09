import launch

launch.run_pip("install diffusers==0.21.0")
if not launch.is_installed("xformers"):
    launch.run_pip("install xformers")
if not launch.is_installed("accelerate"):
    launch.run_pip("install accelerate")
if not launch.is_installed("huggingface_hub"):
    launch.run_pip("install huggingface_hub")
if not launch.is_installed("transformers"):
    launch.run_pip("install transformers")
if not launch.is_installed("stable_diffusion_reference_only"):
    launch.run_pip("install git+https://github.com/aihao2000/stable-diffusion-reference-only.git")
if not launch.is_installed("anime_segmentation"):
    launch.run_pip("install git+https://github.com/aihao2000/anime-segmentation.git")