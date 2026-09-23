# Image Inputs

Use `--imgs` to select an image file or a directory of images. The default input
directory is `testsets/ffhq`. The loader converts images to the model's color
space, spatial dimensions, and numerical range.

Directory inputs are sorted by filename. Observation masks and random seeds
depend on the image index. Use `--image-offset` to preserve indices when
replaying a subset. Image data is excluded from version control.
