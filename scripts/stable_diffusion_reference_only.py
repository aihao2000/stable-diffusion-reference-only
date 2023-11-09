import modules.scripts as scripts
import gradio as gr
import os

from modules import images, script_callbacks
from modules.processing import process_images, Processed
from modules.processing import Processed
from modules.shared import opts, cmd_opts, state


class StableDiffusionReferenceOnlyScript(scripts.Script):
    def title(self):
        return "Stable Diffusion Reference Only"
