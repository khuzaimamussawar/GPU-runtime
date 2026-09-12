import math

import comfy.utils
import node_helpers


class SceneBuilderKrea2StyleEncode10:
    """Krea/Qwen style-reference conditioning with up to 10 optional images.

    This intentionally mirrors ComfyUI's pinned TextEncodeQwenImageEditPlus
    behavior, extending only the number of optional image inputs from 3 to 10.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            },
            "optional": {
                "vae": ("VAE",),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image6": ("IMAGE",),
                "image7": ("IMAGE",),
                "image8": ("IMAGE",),
                "image9": ("IMAGE",),
                "image10": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "SceneBuilder/Krea2"
    DESCRIPTION = "Krea 2 / Qwen style-reference encoder supporting up to 10 images."

    def encode(
        self,
        clip,
        prompt,
        vae=None,
        image1=None,
        image2=None,
        image3=None,
        image4=None,
        image5=None,
        image6=None,
        image7=None,
        image8=None,
        image9=None,
        image10=None,
    ):
        ref_latents = []
        images = [
            image1,
            image2,
            image3,
            image4,
            image5,
            image6,
            image7,
            image8,
            image9,
            image10,
        ]
        images_vl = []

        llama_template = (
            "<|im_start|>system\n"
            "Describe the key features of the input image (color, shape, size, texture, objects, background), "
            "then explain how the user's text instruction should alter or modify the image. Generate a new image "
            "that meets the user's requirements while maintaining consistency with the original input where appropriate."
            "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        )
        image_prompt = ""

        for index, image in enumerate(images, start=1):
            if image is None:
                continue

            samples = image.movedim(-1, 1)

            # Match pinned ComfyUI TextEncodeQwenImageEditPlus vision preprocessing.
            total = int(384 * 384)
            scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
            width = round(samples.shape[3] * scale_by)
            height = round(samples.shape[2] * scale_by)
            scaled = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
            images_vl.append(scaled.movedim(1, -1))

            # Match pinned ComfyUI reference-latent preprocessing.
            if vae is not None:
                total = int(1024 * 1024)
                scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                width = round(samples.shape[3] * scale_by / 8.0) * 8
                height = round(samples.shape[2] * scale_by / 8.0) * 8
                scaled = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
                ref_latents.append(vae.encode(scaled.movedim(1, -1)[:, :, :, :3]))

            image_prompt += f"Picture {index}: <|vision_start|><|image_pad|><|vision_end|>"

        tokens = clip.tokenize(
            image_prompt + prompt,
            images=images_vl,
            llama_template=llama_template,
        )
        conditioning = clip.encode_from_tokens_scheduled(tokens)

        if ref_latents:
            conditioning = node_helpers.conditioning_set_values(
                conditioning,
                {"reference_latents": ref_latents},
                append=True,
            )

        return (conditioning,)


NODE_CLASS_MAPPINGS = {
    "SceneBuilderKrea2StyleEncode10": SceneBuilderKrea2StyleEncode10,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SceneBuilderKrea2StyleEncode10": "SceneBuilder Krea2 Style Encode (10 refs)",
}
