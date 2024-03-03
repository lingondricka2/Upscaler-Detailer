Proof of concept node that upscales the segments inside a detailer node, faster, uses less VRAM and for some reason less noticeably seams. 

Also a modified "Make Tile SEGS" that automatically calculate bbox based on crop region, scale factor and upscale

Upscaling to 16k:

Upscaler + detailer took 3897 seconds and almost went OOM with an Geforce gtx 4090

This node took 1820 seconds

Most code taken from Impact-pack https://github.com/ltdrdata/ComfyUI-Impact-Pack

Upscaler stuff taken from Comfyroll Studio https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes

