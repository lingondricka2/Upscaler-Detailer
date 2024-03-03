import torch
import torchvision
import math
import numpy as np
import nodes
from PIL import Image 
import comfy
import folder_paths
from collections import namedtuple
from comfy_extras.chainner_models import model_loading
from comfy import model_management
from comfy.k_diffusion import sampling as k_diffusion_sampling
from comfy import samplers
from comfy_extras import nodes_custom_sampler
from concurrent.futures import ThreadPoolExecutor

SEG = namedtuple("SEG",
                  ['cropped_image', 'cropped_mask', 'confidence', 'crop_region', 'bbox', 'label', 'control_net_wrapper'],
                  defaults=[None])

def make_crop_region(w, h, bbox, crop_factor, crop_min_size=None):
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[2]
    y2 = bbox[3]

    bbox_w = x2 - x1
    bbox_h = y2 - y1

    crop_w = bbox_w * crop_factor
    crop_h = bbox_h * crop_factor

    if crop_min_size is not None:
        crop_w = max(crop_min_size, crop_w)
        crop_h = max(crop_min_size, crop_h)

    kernel_x = x1 + bbox_w / 2
    kernel_y = y1 + bbox_h / 2

    new_x1 = int(kernel_x - crop_w / 2)
    new_y1 = int(kernel_y - crop_h / 2)

    # make sure position in (w,h)
    new_x1, new_x2 = normalize_region(w, new_x1, crop_w)
    new_y1, new_y2 = normalize_region(h, new_y1, crop_h)

    return [new_x1, new_y1, new_x2, new_y2]

def normalize_region(limit, startp, size):
    if startp < 0:
        new_endp = min(limit, size)
        new_startp = 0
    elif startp + size > limit:
        new_startp = max(0, limit - size)
        new_endp = limit
    else:
        new_startp = startp
        new_endp = min(limit, startp+size)

    return int(new_startp), int(new_endp)

def random_mask_raw(mask, bbox, factor):
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    factor = int(min(w, h) * factor / 4)

    def draw_random_circle(center, radius):
        i, j = center
        for x in range(int(i - radius), int(i + radius)):
            for y in range(int(j - radius), int(j + radius)):
                if np.linalg.norm(np.array([x, y]) - np.array([i, j])) <= radius:
                    mask[x, y] = 1

    def draw_irregular_line(start, end, pivot, is_vertical):
        i = start
        while i < end:
            base_radius = np.random.randint(5, factor)
            radius = int(base_radius)

            if is_vertical:
                draw_random_circle((i, pivot), radius)
            else:
                draw_random_circle((pivot, i), radius)

            i += radius

    def draw_irregular_line_parallel(start, end, pivot, is_vertical):
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = []
            step = (end - start) // 16
            for i in range(start, end, step):
                future = executor.submit(draw_irregular_line, i, min(i + step, end), pivot, is_vertical)
                futures.append(future)

            for future in futures:
                future.result()

    draw_irregular_line_parallel(y1 + factor, y2 - factor, x1 + factor, True)
    draw_irregular_line_parallel(y1 + factor, y2 - factor, x2 - factor, True)
    draw_irregular_line_parallel(x1 + factor, x2 - factor, y1 + factor, False)
    draw_irregular_line_parallel(x1 + factor, x2 - factor, y2 - factor, False)

    mask[y1 + factor:y2 - factor, x1 + factor:x2 - factor] = 1.0


def random_mask(mask, bbox, factor, size=128):
    small_mask = np.zeros((size, size)).astype(np.float32)
    random_mask_raw(small_mask, (0, 0, size, size), factor)

    x1, y1, x2, y2 = bbox
    small_mask = torch.tensor(small_mask).unsqueeze(0).unsqueeze(0)
    bbox_mask = torch.nn.functional.interpolate(small_mask, size=(y2 - y1, x2 - x1), mode='bilinear', align_corners=False)
    bbox_mask = bbox_mask.squeeze(0).squeeze(0)
    mask[y1:y2, x1:x2] = bbox_mask


def adaptive_mask_paste(dest_mask, src_mask, bbox):
    x1, y1, x2, y2 = bbox
    small_mask = torch.tensor(src_mask).unsqueeze(0).unsqueeze(0)
    bbox_mask = torch.nn.functional.interpolate(small_mask, size=(y2 - y1, x2 - x1), mode='bilinear', align_corners=False)
    bbox_mask = bbox_mask.squeeze(0).squeeze(0)
    dest_mask[y1:y2, x1:x2] = bbox_mask

def calculate_sigmas(model, sampler, scheduler, steps):
    discard_penultimate_sigma = False
    if sampler in ['dpm_2', 'dpm_2_ancestral', 'uni_pc', 'uni_pc_bh2']:
        steps += 1
        discard_penultimate_sigma = True

    sigmas = samplers.calculate_sigmas_scheduler(model.model, scheduler, steps)

    if discard_penultimate_sigma:
        sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])
    return sigmas


def get_noise_sampler(x, cpu, total_sigmas, **kwargs):
    if 'extra_args' in kwargs and 'seed' in kwargs['extra_args']:
        sigma_min, sigma_max = total_sigmas[total_sigmas > 0].min(), total_sigmas.max()
        seed = kwargs['extra_args'].get("seed", None)
        return k_diffusion_sampling.BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=cpu)
    return None


def ksampler(sampler_name, total_sigmas, extra_options={}, inpaint_options={}):
    if sampler_name == "dpmpp_sde":
        def sample_dpmpp_sde(model, x, sigmas, **kwargs):
            noise_sampler = get_noise_sampler(x, True, total_sigmas, **kwargs)
            if noise_sampler is not None:
                kwargs['noise_sampler'] = noise_sampler

            return k_diffusion_sampling.sample_dpmpp_sde(model, x, sigmas, **kwargs)

        sampler_function = sample_dpmpp_sde

    elif sampler_name == "dpmpp_sde_gpu":
        def sample_dpmpp_sde(model, x, sigmas, **kwargs):
            noise_sampler = get_noise_sampler(x, False, total_sigmas, **kwargs)
            if noise_sampler is not None:
                kwargs['noise_sampler'] = noise_sampler

            return k_diffusion_sampling.sample_dpmpp_sde_gpu(model, x, sigmas, **kwargs)

        sampler_function = sample_dpmpp_sde

    elif sampler_name == "dpmpp_2m_sde":
        def sample_dpmpp_sde(model, x, sigmas, **kwargs):
            noise_sampler = get_noise_sampler(x, True, total_sigmas, **kwargs)
            if noise_sampler is not None:
                kwargs['noise_sampler'] = noise_sampler

            return k_diffusion_sampling.sample_dpmpp_2m_sde(model, x, sigmas, **kwargs)

        sampler_function = sample_dpmpp_sde

    elif sampler_name == "dpmpp_2m_sde_gpu":
        def sample_dpmpp_sde(model, x, sigmas, **kwargs):
            noise_sampler = get_noise_sampler(x, False, total_sigmas, **kwargs)
            if noise_sampler is not None:
                kwargs['noise_sampler'] = noise_sampler

            return k_diffusion_sampling.sample_dpmpp_2m_sde_gpu(model, x, sigmas, **kwargs)

        sampler_function = sample_dpmpp_sde

    elif sampler_name == "dpmpp_3m_sde":
        def sample_dpmpp_sde(model, x, sigmas, **kwargs):
            noise_sampler = get_noise_sampler(x, True, total_sigmas, **kwargs)
            if noise_sampler is not None:
                kwargs['noise_sampler'] = noise_sampler

            return k_diffusion_sampling.sample_dpmpp_2m_sde(model, x, sigmas, **kwargs)

        sampler_function = sample_dpmpp_sde

    elif sampler_name == "dpmpp_3m_sde_gpu":
        def sample_dpmpp_sde(model, x, sigmas, **kwargs):
            noise_sampler = get_noise_sampler(x, False, total_sigmas, **kwargs)
            if noise_sampler is not None:
                kwargs['noise_sampler'] = noise_sampler

            return k_diffusion_sampling.sample_dpmpp_2m_sde_gpu(model, x, sigmas, **kwargs)

        sampler_function = sample_dpmpp_sde
    else:
        return samplers.ksampler(sampler_name, extra_options, inpaint_options)

    return samplers.KSAMPLER(sampler_function, extra_options, inpaint_options)


def separated_sample(model, add_noise, seed, steps, cfg, sampler_name, scheduler, positive, negative,
                     latent_image, start_at_step, end_at_step, return_with_leftover_noise, sigma_ratio=1.0):
    total_sigmas = calculate_sigmas(model, sampler_name, scheduler, steps)

    sigmas = total_sigmas[start_at_step:end_at_step+1] * sigma_ratio
    impact_sampler = ksampler(sampler_name, total_sigmas)

    if len(sigmas) == 0 or (len(sigmas) == 1 and sigmas[0] == 0):
        return latent_image
    
    res = nodes_custom_sampler.SamplerCustom().sample(model, add_noise, seed, cfg, positive, negative, impact_sampler, sigmas, latent_image)

    if return_with_leftover_noise:
        return res[0]
    else:
        return res[1]


def ksampler_wrapper(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise,
                     refiner_ratio=None, refiner_model=None, refiner_clip=None, refiner_positive=None, refiner_negative=None):

    if refiner_ratio is None or refiner_model is None or refiner_clip is None or refiner_positive is None or refiner_negative is None:
        refined_latent = nodes.KSampler().sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise)[0]
    else:
        advanced_steps = math.floor(steps / denoise)
        start_at_step = advanced_steps - steps
        end_at_step = start_at_step + math.floor(steps * (1.0 - refiner_ratio))

        print(f"pre: {start_at_step} .. {end_at_step} / {advanced_steps}")
        temp_latent = separated_sample(model, True, seed, advanced_steps, cfg, sampler_name, scheduler,
                                       positive, negative, latent_image, start_at_step, end_at_step, True)

        if 'noise_mask' in latent_image:
            # noise_latent = \
            #     impact_sampling.separated_sample(refiner_model, "enable", seed, advanced_steps, cfg, sampler_name,
            #                                      scheduler, refiner_positive, refiner_negative, latent_image, end_at_step,
            #                                      end_at_step, "enable")

            latent_compositor = nodes.NODE_CLASS_MAPPINGS['LatentCompositeMasked']()
            temp_latent = latent_compositor.composite(latent_image, temp_latent, 0, 0, False, latent_image['noise_mask'])[0]

        print(f"post: {end_at_step} .. {advanced_steps + 1} / {advanced_steps}")
        refined_latent = separated_sample(refiner_model, False, seed, advanced_steps, cfg, sampler_name, scheduler,
                                          refiner_positive, refiner_negative, temp_latent, end_at_step, advanced_steps + 1, False)

    return refined_latent

def tensor_convert_rgba(image, prefer_copy=True):
    """Assumes NHWC format tensor with 1, 3 or 4 channels."""
    _tensor_check_image(image)
    n_channel = image.shape[-1]
    if n_channel == 4:
        return image

    if n_channel == 3:
        alpha = torch.ones((*image.shape[:-1], 1))
        return torch.cat((image, alpha), axis=-1)

    if n_channel == 1:
        if prefer_copy:
            image = image.repeat(1, -1, -1, 4)
        else:
            image = image.expand(1, -1, -1, 3)
        return image

    # NOTE: Similar error message as in PIL, for easier googling :P
    raise ValueError(f"illegal conversion (channels: {n_channel} -> 4)")


def tensor_convert_rgb(image, prefer_copy=True):
    """Assumes NHWC format tensor with 1, 3 or 4 channels."""
    _tensor_check_image(image)
    n_channel = image.shape[-1]
    if n_channel == 3:
        return image

    if n_channel == 4:
        image = image[..., :3]
        if prefer_copy:
            image = image.copy()
        return image

    if n_channel == 1:
        if prefer_copy:
            image = image.repeat(1, -1, -1, 4)
        else:
            image = image.expand(1, -1, -1, 3)
        return image

    # NOTE: Same error message as in PIL, for easier googling :P
    raise ValueError(f"illegal conversion (channels: {n_channel} -> 3)")


def general_tensor_resize(image, w: int, h: int):
    _tensor_check_image(image)
    image = image.permute(0, 3, 1, 2)
    image = torch.nn.functional.interpolate(image, size=(h, w), mode="bilinear")
    image = image.permute(0, 2, 3, 1)
    return image


# TODO: Sadly, we need LANCZOS
LANCZOS = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
def tensor_resize(image, w: int, h: int):
    _tensor_check_image(image)
    if image.shape[3] >= 3:
        image = tensor2pil(image)
        scaled_image = image.resize((w, h), resample=LANCZOS)
        return pil2tensor(scaled_image)
    else:
        return general_tensor_resize(image, w, h)


def tensor_get_size(image):
    """Mimicking `PIL.Image.size`"""
    _tensor_check_image(image)
    _, h, w, _ = image.shape
    return (w, h)


def tensor2pil(image):
    _tensor_check_image(image)
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(0), 0, 255).astype(np.uint8))

def tensor2pil_upscaler(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def to_tensor(image):
    if isinstance(image, Image.Image):
        return torch.from_numpy(np.array(image))
    if isinstance(image, torch.Tensor):
        return image
    if isinstance(image, np.ndarray):
        return torch.from_numpy(image)
    raise ValueError(f"Cannot convert {type(image)} to torch.Tensor")

def tensor_putalpha(image, mask):
    _tensor_check_image(image)
    _tensor_check_mask(mask)
    image[..., -1] = mask[..., 0]


def _tensor_check_image(image):
    if image.ndim != 4:
        raise ValueError(f"Expected NHWC tensor, but found {image.ndim} dimensions")
    if image.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Expected 1, 3 or 4 channels for image, but found {image.shape[-1]} channels")
    return


def _tensor_check_mask(mask):
    if mask.ndim != 4:
        raise ValueError(f"Expected NHWC tensor, but found {mask.ndim} dimensions")
    if mask.shape[-1] != 1:
        raise ValueError(f"Expected 1 channel for mask, but found {mask.shape[-1]} channels")
    return

def tensor_paste(image1, image2, left_top, mask):
    """Mask and image2 has to be the same size"""
    _tensor_check_image(image1)
    _tensor_check_image(image2)
    _tensor_check_mask(mask)
    if image2.shape[1:3] != mask.shape[1:3]:
        raise ValueError(f"Inconsistent size: Image ({image2.shape[1:3]}) != Mask ({mask.shape[1:3]})")

    x, y = left_top
    _, h1, w1, _ = image1.shape
    _, h2, w2, _ = image2.shape

    # calculate image patch size
    w = min(w1, x + w2) - x
    h = min(h1, y + h2) - y

    # If the patch is out of bound, nothing to do!
    if w <= 0 or h <= 0:
        return

    mask = mask[:, :h, :w, :]
    image1[:, y:y+h, x:x+w, :] = (
        (1 - mask) * image1[:, y:y+h, x:x+w, :] +
        mask * image2[:, :h, :w, :]
    )
    return

def tensor_gaussian_blur_mask(mask, kernel_size, sigma=10.0):
    """Return NHWC torch.Tenser from ndim == 2 or 4 `np.ndarray` or `torch.Tensor`"""
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)

    if mask.ndim == 2:
        mask = mask[None, ..., None]
    elif mask.ndim == 3:
        mask = mask[..., None]

    _tensor_check_mask(mask)

    if kernel_size <= 0:
        return mask

    kernel_size = kernel_size*2+1

    shortest = min(mask.shape[1], mask.shape[2])
    if shortest <= kernel_size:
        kernel_size = int(shortest/2)
        if kernel_size % 2 == 0:
            kernel_size += 1
        if kernel_size < 3:
            return mask  # skip feathering

    prev_device = mask.device
    device = comfy.model_management.get_torch_device()
    mask.to(device)

    # apply gaussian blur
    mask = mask[:, None, ..., 0]
    blurred_mask = torchvision.transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)(mask)
    blurred_mask = blurred_mask[:, 0, ..., None]

    blurred_mask.to(prev_device)

    return blurred_mask

def crop_ndarray4(npimg, crop_region):
    x1 = crop_region[0]
    y1 = crop_region[1]
    x2 = crop_region[2]
    y2 = crop_region[3]

    cropped = npimg[:, y1:y2, x1:x2, :]

    return cropped

def to_latent_image(pixels, vae):
    x = pixels.shape[1]
    y = pixels.shape[2]
    if pixels.shape[1] != x or pixels.shape[2] != y:
        pixels = pixels[:, :x, :y, :]
    pixels = nodes.VAEEncode.vae_encode_crop_pixels(pixels)
    t = vae.encode(pixels[:, :, :, :3])
    return {"samples": t}

def segs_scale_match(segs, target_shape):
    h = segs[0][0]
    w = segs[0][1]

    th = target_shape[1]
    tw = target_shape[2]

    if (h == th and w == tw) or h == 0 or w == 0:
        return segs

    rh = th / h
    rw = tw / w

    new_segs = []
    for seg in segs[1]:
        cropped_image = seg.cropped_image
        cropped_mask = seg.cropped_mask
        x1, y1, x2, y2 = seg.crop_region
        bx1, by1, bx2, by2 = seg.bbox

        crop_region = int(x1*rw), int(y1*rw), int(x2*rh), int(y2*rh)
        bbox = int(bx1*rw), int(by1*rw), int(bx2*rh), int(by2*rh)
        new_w = crop_region[2] - crop_region[0]
        new_h = crop_region[3] - crop_region[1]

        cropped_mask = torch.from_numpy(cropped_mask)
        cropped_mask = torch.nn.functional.interpolate(cropped_mask.unsqueeze(0).unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False)
        cropped_mask = cropped_mask.squeeze(0).squeeze(0).numpy()

        if cropped_image is not None:
            cropped_image = tensor_resize(cropped_image if isinstance(cropped_image, torch.Tensor) else torch.from_numpy(cropped_image), new_w, new_h)
            cropped_image = cropped_image.numpy()

        new_seg = SEG(cropped_image, cropped_mask, seg.confidence, crop_region, bbox, seg.label, seg.control_net_wrapper)
        new_segs.append(new_seg)

    return (th, tw), new_segs


def load_model(model_name):
    model_path = folder_paths.get_full_path("upscale_models", model_name)
    sd = comfy.utils.load_torch_file(model_path, safe_load=True)
    if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
        sd = comfy.utils.state_dict_prefix_replace(sd, {"module.":""})
    out = model_loading.load_state_dict(sd).eval()
    return out

def upscale_with_model(upscale_model, image):
    device = model_management.get_torch_device()
    upscale_model.to(device)
    in_img = image.movedim(-1,-3).to(device)
    free_memory = model_management.get_free_memory(device)

    tile = 512
    overlap = 32

    oom = True
    while oom:
        try:
            steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap)
            pbar = comfy.utils.ProgressBar(steps)
            s = comfy.utils.tiled_scale(in_img, lambda a: upscale_model(a), tile_x=tile, tile_y=tile, overlap=overlap, upscale_amount=upscale_model.scale, pbar=pbar)
            oom = False
        except model_management.OOM_EXCEPTION as e:
            tile //= 2
            if tile < 128:
                raise e

    upscale_model.cpu()
    s = torch.clamp(s.movedim(-3,-1), min=0, max=1.0)
    return s    

def apply_resize_image(image: Image.Image, original_width, original_height, rounding_modulus, mode='scale', supersample='true', factor: int = 2, width: int = 1024, height: int = 1024, resample='bicubic'): 

    # Calculate the new width and height based on the given mode and parameters
    if mode == 'rescale':
        new_width, new_height = int(original_width * factor), int(original_height * factor)               
    else:
        m = rounding_modulus
        original_ratio = original_height / original_width
        height = int(width * original_ratio)
        
        new_width = width if width % m == 0 else width + (m - width % m)
        new_height = height if height % m == 0 else height + (m - height % m)

    # Define a dictionary of resampling filters
    resample_filters = {'nearest': 0, 'bilinear': 2, 'bicubic': 3, 'lanczos': 1}
    
    # Apply supersample
    if supersample == 'true':
        image = image.resize((new_width * 8, new_height * 8), resample=Image.Resampling(resample_filters[resample]))

    # Resize the image using the given resampling filter
    resized_image = image.resize((new_width, new_height), resample=Image.Resampling(resample_filters[resample]))
    
    return resized_image

def upscaler(image, upscale_model, rescale_factor, resampling_method, supersample, rounding_modulus):
    up_model = load_model(upscale_model)
    up_image = upscale_with_model(up_model, image)  
    pil_img = tensor2pil_upscaler(image)
    original_width, original_height = pil_img.size
    scaled_image = pil2tensor(apply_resize_image(tensor2pil_upscaler(up_image), original_width, original_height, rounding_modulus, 'rescale', 
                                                    supersample, rescale_factor, 1024, resampling_method))
    return scaled_image

def enhance_detail_modified(image, model, clip, vae, upscale_model, rescale_factor, resampling_method, supersample, rounding_modulus, 
                            bbox, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise, noise_mask, 
                            control_net_wrapper=None, noise_mask_feather=0):

    if noise_mask is not None:
        noise_mask = tensor_gaussian_blur_mask(noise_mask, noise_mask_feather)
        noise_mask = noise_mask.squeeze(3)


    h = image.shape[1]
    w = image.shape[2]

    bbox_h = bbox[3] - bbox[1]
    bbox_w = bbox[2] - bbox[0]

    # Skip processing if the detected bbox is already larger than the guide_size


  
    upscale = rescale_factor

    new_w = int(w * upscale)
    new_h = int(h * upscale)

    print(f"Detailer: segment upscale for ({bbox_w, bbox_h}) | crop region {w, h} x {upscale} -> {new_w, new_h}")

    upscaled_image = upscaler(image, upscale_model, rescale_factor, resampling_method, supersample, rounding_modulus)
    #upscaled_image = tensor_resize(image, new_w, new_h)

  
    if control_net_wrapper is not None:
        positive, negative, _ = control_net_wrapper.apply(positive, negative, upscaled_image, noise_mask)

    # prepare mask
    #if noise_mask is not None and inpaint_model:
    #    positive, negative, latent_image = nodes.InpaintModelConditioning().encode(positive, negative, upscaled_image, vae, noise_mask)
    #else:
    latent_image = to_latent_image(upscaled_image, vae)
    if noise_mask is not None:
        latent_image['noise_mask'] = noise_mask


    refined_latent = latent_image

    # ksampler
    refined_latent = ksampler_wrapper(model, seed, steps, cfg, sampler_name, scheduler, positive, negative,
                                                        refined_latent, denoise)


    # non-latent downscale - latent downscale cause bad quality
    refined_image = vae.decode(refined_latent['samples'])

    # downscale
    #refined_image = tensor_resize(refined_image, w, h)

    # prevent mixing of device
    refined_image = refined_image.cpu()

    # don't convert to latent - latent break image
    # preserving pil is much better
    return refined_image

class UpscalerDetailer:

    @classmethod
    def INPUT_TYPES(s):

        resampling_methods = ["lanczos", "nearest", "bilinear", "bicubic"]

        return {"required": {
                    "image": ("IMAGE", ),
                    "segs": ("SEGS", ),
                    "model": ("MODEL",),
                    "clip": ("CLIP",),
                    "vae": ("VAE",),
                    "upscale_model": (folder_paths.get_filename_list("upscale_models"), ),
                    "rescale_factor": ("DATA",),
                    "resampling_method": (resampling_methods,),                     
                    "supersample": (["true", "false"],),   
                    "rounding_modulus": ("INT", {"default": 8, "min": 8, "max": 1024, "step": 8}),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                    "positive": ("CONDITIONING",),
                    "negative": ("CONDITIONING",),
                    "denoise": ("FLOAT", {"default": 0.5, "min": 0.0001, "max": 1.0, "step": 0.01}),
                    "feather": ("INT", {"default": 5, "min": 0, "max": 100, "step": 1}),
                    "noise_mask": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                   },
                "optional": {
                    "noise_mask_feather": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                   }
                }

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "do_detail"

    CATEGORY = "ImpactPack/Detailer"


    def do_detail(self, image, segs, model, clip, vae, upscale_model, rescale_factor, resampling_method, supersample, rounding_modulus, 
                  seed, steps, cfg, sampler_name, scheduler,
                  positive, negative, denoise, feather, noise_mask, noise_mask_feather):

        image = image.clone()
        new_image = image.clone()

        h = image.shape[1]
        w = image.shape[2]

        upscale = rescale_factor

        new_w = int(w * upscale)
        new_h = int(h * upscale)


        new_image = tensor_resize(new_image, new_w, new_h)

        segs = segs_scale_match(segs, image.shape)
   

     
        ordered_segs = segs[1]

        for i, seg in enumerate(ordered_segs):
            cropped_image = seg.cropped_image if seg.cropped_image is not None \
                                              else crop_ndarray4(image.numpy(), seg.crop_region)
            cropped_image = to_tensor(cropped_image)
            mask = to_tensor(seg.cropped_mask)
            mask = tensor_gaussian_blur_mask(mask, feather)

            is_mask_all_zeros = (seg.cropped_mask == 0).all().item()
            if is_mask_all_zeros:
                print(f"Detailer: segment skip [empty mask]")
                continue

            if noise_mask:
                cropped_mask = seg.cropped_mask
            else:
                cropped_mask = None

            seg_seed = seed + i

            enhanced_image = enhance_detail_modified(cropped_image, model, clip, vae, upscale_model, rescale_factor, resampling_method, 
                                                     supersample, rounding_modulus, seg.bbox, seg_seed, steps, cfg, sampler_name, 
                                                     scheduler, positive, negative, denoise, cropped_mask, 
                                                     control_net_wrapper=seg.control_net_wrapper, noise_mask_feather=noise_mask_feather)
            if not (enhanced_image is None):
                new_image = new_image.cpu()
                enhanced_image = enhanced_image.cpu()
                seg.crop_region[0] = int(seg.crop_region[0] * upscale)
                seg.crop_region[1] = int(seg.crop_region[1] * upscale)
                h = enhanced_image.shape[1]
                w = enhanced_image.shape[2]
                mask = tensor_resize(mask, w, h)

                tensor_paste(new_image, enhanced_image, (seg.crop_region[0], seg.crop_region[1]), mask)
        enhanced_img = tensor_convert_rgb(new_image)

        return (enhanced_img, )
    
class MakeTileSEGSForUpscaler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "image": ("IMAGE", ),
                     "upscale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 100, "step": 0.1}),
                     "tile_size": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
                     "crop_factor": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 10, "step": 0.1}),
                     "mask_irregularity": ("FLOAT", {"default": 0, "min": 0, "max": 1.0, "step": 0.01}),
                     "irregular_mask_mode": (["Reuse fast", "Reuse quality", "All random fast", "All random quality"],)
                    },
                }

    RETURN_TYPES = ("SEGS", "DATA", ) 

    FUNCTION = "doit"

    CATEGORY = "ImpactPack/__for_testing"

    def doit(self, image, upscale, tile_size, crop_factor, mask_irregularity=0, irregular_mask_mode="Reuse fast"):
        _, ih, iw, _ = image.size()
        bbox_size = int((tile_size / crop_factor) / upscale)
        mask_cache = None
        mask_quality = 512
        if mask_irregularity > 0:
            if irregular_mask_mode == "Reuse fast":
                mask_quality = 128
                mask_cache = np.zeros((128, 128)).astype(np.float32)
                random_mask(mask_cache, (0, 0, 128, 128), factor=mask_irregularity, size=mask_quality)
            elif irregular_mask_mode == "Reuse quality":
                mask_quality = 512
                mask_cache = np.zeros((512, 512)).astype(np.float32)
                random_mask(mask_cache, (0, 0, 512, 512), factor=mask_irregularity, size=mask_quality)
            elif irregular_mask_mode == "All random fast":
                mask_quality = 512

  
        exclusion_mask = None
        start_x = 0
        start_y = 0
        h, w = ih, iw
        and_mask = None

        # calculate tile factors
        if bbox_size > h or bbox_size > w:
            new_bbox_size = min(bbox_size, min(w, h))
            print(f"[MaskTileSEGS] bbox_size is greater than resolution (value changed: {bbox_size} => {new_bbox_size}")
            bbox_size = new_bbox_size

        n_horizontal = int(w / bbox_size)
        n_vertical = int(h / bbox_size)
        
        while (((bbox_size - (bbox_size * mask_irregularity * 0.2)) * n_horizontal) - w) < 1:
            n_horizontal += 1
            
        while (((bbox_size - (bbox_size * mask_irregularity * 0.2)) * n_vertical) - w) < 1:
            n_vertical += 1

        w_overlap_sum = (bbox_size * n_horizontal) - w

        w_overlap_size = 0 if n_horizontal == 1 else int(w_overlap_sum/(n_horizontal-1))

        h_overlap_sum = (bbox_size * n_vertical) - h

        h_overlap_size = 0 if n_vertical == 1 else int(h_overlap_sum/(n_vertical-1))

        new_segs = []

        y = start_y
        for j in range(0, n_vertical):
            x = start_x
            for i in range(0, n_horizontal):
                x1 = x
                y1 = y

                if x+bbox_size < iw-1:
                    x2 = x+bbox_size
                else:
                    x2 = iw
                    x1 = iw-bbox_size

                if y+bbox_size < ih-1:
                    y2 = y+bbox_size
                else:
                    y2 = ih
                    y1 = ih-bbox_size

                bbox = x1, y1, x2, y2
                crop_region = make_crop_region(iw, ih, bbox, crop_factor)
                cx1, cy1, cx2, cy2 = crop_region

                mask = np.zeros((cy2 - cy1, cx2 - cx1)).astype(np.float32)

                rel_left = x1 - cx1
                rel_top = y1 - cy1
                rel_right = x2 - cx1
                rel_bot = y2 - cy1

                if mask_irregularity > 0:
                    if mask_cache is not None:
                        adaptive_mask_paste(mask, mask_cache, (rel_left, rel_top, rel_right, rel_bot))
                    else:
                        random_mask(mask, (rel_left, rel_top, rel_right, rel_bot), factor=mask_irregularity, size=mask_quality)

                    # corner filling
                    if rel_left == 0:
                        pad = int((x2 - x1) / 8)
                        mask[rel_top:rel_bot, :pad] = 1.0

                    if rel_top == 0:
                        pad = int((y2 - y1) / 8)
                        mask[:pad, rel_left:rel_right] = 1.0

                    if rel_right == mask.shape[1]:
                        pad = int((x2 - x1) / 8)
                        mask[rel_top:rel_bot, -pad:] = 1.0

                    if rel_bot == mask.shape[0]:
                        pad = int((y2 - y1) / 8)
                        mask[-pad:, rel_left:rel_right] = 1.0
                else:
                    mask[rel_top:rel_bot, rel_left:rel_right] = 1.0

                mask = torch.tensor(mask)

                if exclusion_mask is not None:
                    exclusion_mask_cropped = exclusion_mask[cy1:cy2, cx1:cx2]
                    mask[exclusion_mask_cropped != 0] = 0.0

                if and_mask is not None:
                    and_mask_cropped = and_mask[cy1:cy2, cx1:cx2]
                    mask[and_mask_cropped == 0] = 0.0

                is_mask_zero = torch.all(mask == 0.0).item()

                if not is_mask_zero:
                    item = SEG(None, mask.numpy(), 1.0, crop_region, bbox, "", None)
                    new_segs.append(item)

                x += bbox_size - w_overlap_size
            y += bbox_size - h_overlap_size

        res = (ih, iw), new_segs  # segs
        return (res, upscale, )

    
NODE_CLASS_MAPPINGS = {
    "UpscalerDetailer": UpscalerDetailer,
    "MakeTileSEGSForUpscaler": MakeTileSEGSForUpscaler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UpscalerDetailer": "UpscalerDetailer",
    "MakeTileSEGSForUpscaler": "MakeTileSEGSForUpscaler"
}


