import torch
import torchvision


def debug_dino_dataloader(loader):
    for i, samples in enumerate(loader):
        images, target_bbox, targets = samples
        images, mask = images.decompose()
        assert images.max() >= 0.0
        assert images.min() <= 1.0

        debug_index = 0
        torchvision.utils.save_image(
            images[debug_index], f"example_images/img_{debug_index}.png"
        )
        torchvision.utils.save_image(
            mask[debug_index] * 1.0,
            f"example_images/mask_{debug_index}.png",
        )
        img_uint8 = torchvision.io.read_image(f"example_images/img_{debug_index}.png")
        shape = target_bbox[debug_index]["size"]

        # xc, xy, w, h convert to xmin, ymin, xmax, ymax
        boxes = target_bbox[debug_index]["boxes"]

        if boxes.shape[0] > 0:
            boxes = torchvision.ops.box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
            boxes[:, ::2] = boxes[:, ::2] * shape[1]
            boxes[:, 1::2] = boxes[:, 1::2] * shape[0]

            img_with_boxes = torchvision.utils.draw_bounding_boxes(
                img_uint8, boxes=boxes, colors="red"
            )
        else:
            img_with_boxes = img_uint8

        torchvision.utils.save_image(
            img_with_boxes / 255,
            f"example_images/img_{debug_index}_with_bbox.png",
        )
        import pdb

        pdb.set_trace()
