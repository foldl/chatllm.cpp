# Multimodal

## Dependencies

To keep things simple, external tools are used for accessing and processing multimedia data.
These tools should be available in OS's searching paths.

* **[ffmpeg](https://ffmpeg.org/)**: v7.0.2 or compatible

    `ffmpeg` and `ffplay` are used for video/audio IO.

* **[ImageMagick](https://www.imagemagick.org/)**: v7.1.1 or compatible

    `magick` is used for image IO.

## Input multimedia files

Use `--multimedia_file_tags` to specify a pair of tags, for example:

```
--multimedia_file_tags {{ }}
```

Then an `image`, `audio`, or `video` can be embedded in prompt like this:

```
{{tag:/path/to/an/image/file}}
```

where `tag` is `image`, `audio`, or `video` respectively.

Take Fuyu model as an example:

```
main -m /path/to/fuyu-8b.bin  -p "{{image:path/to/bus.png}}Generate a coco-style caption." --multimedia_file_tags {{ }} -ngl all --max_length 4000
```