Proof of concept node that upscales the segments inside a detailer node, helps with VRAM issues using controlnet preprocessors with large images.

Most code taken from Impact-pack https://github.com/ltdrdata/ComfyUI-Impact-Pack

Upscaler stuff taken from Comfyroll Studio https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes

Does not work with SD 1.5, get error:

```
ERROR:root:!!! Exception during processing !!!
ERROR:root:Traceback (most recent call last):
  File "C:\Users\lingo\Desktop\ComfyUI\execution.py", line 152, in recursive_execute
    output_data, output_ui = get_output_data(obj, input_data_all)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\lingo\Desktop\ComfyUI\execution.py", line 82, in get_output_data
    return_values = map_node_over_list(obj, input_data_all, obj.FUNCTION, allow_interrupt=True)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\lingo\Desktop\ComfyUI\execution.py", line 75, in map_node_over_list
    results.append(getattr(obj, func)(**slice_dict(input_data_all, i)))
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\lingo\Desktop\ComfyUI\custom_nodes\Upscaler_Detailer\upscaler_detailer.py", line 600, in do_detail
    enhanced_image = enhance_detail_modified(cropped_image, model, clip, vae, upscale_model, rescale_factor, resampling_method,
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\lingo\Desktop\ComfyUI\custom_nodes\Upscaler_Detailer\upscaler_detailer.py", line 481, in enhance_detail_modified
    upscaled_image = upscaler(image, upscale_model, rescale_factor, resampling_method, supersample, rounding_modulus)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\lingo\Desktop\ComfyUI\custom_nodes\Upscaler_Detailer\upscaler_detailer.py", line 449, in upscaler
    pil_img = tensor2pil(image)
              ^^^^^^^^^^^^^^^^^
  File "C:\Users\lingo\Desktop\ComfyUI\custom_nodes\Upscaler_Detailer\upscaler_detailer.py", line 229, in tensor2pil
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(0), 0, 255).astype(np.uint8))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\lingo\AppData\Local\Programs\Python\Python311\Lib\site-packages\PIL\Image.py", line 3112, in fromarray
    return frombuffer(mode, size, obj, "raw", rawmode, 0, 1)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\lingo\AppData\Local\Programs\Python\Python311\Lib\site-packages\PIL\Image.py", line 3028, in frombuffer
    return frombytes(mode, size, data, decoder_name, args)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\lingo\AppData\Local\Programs\Python\Python311\Lib\site-packages\PIL\Image.py", line 2970, in frombytes
    im.frombytes(data, decoder_name, args)
  File "C:\Users\lingo\AppData\Local\Programs\Python\Python311\Lib\site-packages\PIL\Image.py", line 821, in frombytes
    d.setimage(self.im)
ValueError: tile cannot extend outside image
```