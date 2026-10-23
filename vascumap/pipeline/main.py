import os
import argparse
import glob
import numpy as np
import tifffile as tif
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from aicsimageio import AICSImage

from .data import get_stack_from_lif, get_stack_from_aics
from .preprocessing import crop_stack
from .prediction import load_pix2pix, load_segmentation_model, predict_pix2pix, predict_mask, predict_mask_ortho
from .utils import contrast, scale, resize_dask, process_vessel_mask


def _find_channel(channel_names, keys):
    """Index of the first channel whose name contains one of keys (case-insensitive), else None."""
    for i, name in enumerate(channel_names):
        if any(k.lower() in str(name).lower() for k in keys):
            return i
    return None


def _iter_stacks(image_path, fluo_channel, fluo_index):
    """
    Yield (filename, stack, channel_names) for every 3D stack (scene) of a microscopy file.

    LIF files are read with readlif (all channels). With a fluorescence channel requested,
    any format readable by aicsimageio (e.g. .nd2, .czi, .lif) is read scene by scene and
    only that channel is returned, as a (Z, Y, X, 1) stack. The scene stays selected until
    the caller asks for the next one, so each loader must run before the generator advances.
    """
    stem = Path(image_path).stem
    fluo_mode = fluo_channel is not None or fluo_index is not None

    if image_path.lower().endswith('.lif') and not fluo_mode:
        from readlif.reader import LifFile
        reader = LifFile(image_path)
        channel_names = list(AICSImage(image_path).channel_names or [])
        for stack_index in range(reader.num_images):
            img_obj = reader.get_image(stack_index)
            filename = f"{stem}_{img_obj.name}".replace(' ', '_').replace('/', '_')
            if len(img_obj.dims_n) != 3:
                print(f'{filename} is not a 3D stack. Skipped')
                continue
            yield filename, (get_stack_from_lif, dict(lif_file=image_path, stack_index=stack_index)), channel_names
        return

    img = AICSImage(image_path)
    for scene_id in img.scenes:
        img.set_scene(scene_id)
        filename = f"{stem}_{scene_id}".replace(' ', '_').replace('/', '_')
        if img.dims.Z <= 1:
            print(f'{filename} is not a 3D stack. Skipped')
            continue
        channel_names = list(img.channel_names or [])
        index = fluo_index if fluo_index is not None else _find_channel(channel_names, [fluo_channel])
        if index is None:
            print(f'{filename}: no channel matching "{fluo_channel}" in {channel_names}. Skipped')
            continue
        yield filename, (get_stack_from_aics, dict(img=img, channel_index=index)), channel_names


def run(input_dir, pattern='*.lif', ortho=True, bf_index=None, fluo_channel=None, fluo_index=None,
        crop_profile='dorota', crop_profile_dir='crop_profiles', z_range=200, save_all_stacks=False,
        crop=True, model_p2p_path=None, model_smp_path=None):
    """
    Run the inference pipeline on every microscopy file of a directory.

    Steps:
    1. Load each 3D stack and resample it to the reference voxel size (5 x 2 x 2 um).
    2. (Optional) Auto-focus and crop the stack using a reference crop profile.
    3. Brightfield input: translate brightfield to fluorescence with Pix2Pix 3D.
       Fluorescence input: segment the fluorescence channel directly.
    4. Segment the (predicted) fluorescence volume with the 2D U-Net (2.5D if ortho).
    5. Save the isotropic 2 um vessel mask, an overview figure and, optionally,
       the intermediate stacks next to the input files.

    Args:
        input_dir (str): Directory containing the microscopy files.
        pattern (str, optional): Glob pattern of the files to process. Defaults to '*.lif'.
        ortho (bool, optional): Average axial, coronal and sagittal predictions (2.5D). Defaults to True.
        bf_index (int, optional): Index of the brightfield channel. Defaults to None (found by
            name: 'BF' or 'Gray', else channel 0).
        fluo_channel (str, optional): Name (substring) of a fluorescence vessel channel, e.g. 'GFP'.
            When given, Pix2Pix is skipped. Defaults to None.
        fluo_index (int, optional): Index of the fluorescence vessel channel; same effect as
            fluo_channel. Defaults to None.
        crop_profile (str, optional): Name of the crop profile for registration. Defaults to 'dorota'.
        crop_profile_dir (str, optional): Folder holding the crop profiles. Defaults to 'crop_profiles'.
        z_range (int, optional): Z-range (um) kept around the focal surface when cropping. Defaults to 200.
        save_all_stacks (bool, optional): Save intermediate stacks (BF, prediction, probability). Defaults to False.
        crop (bool, optional): Whether to apply cropping. Defaults to True.
        model_p2p_path (str, optional): Path to the Pix2Pix checkpoint (brightfield input only). Defaults to None.
        model_smp_path (str): Path to the segmentation checkpoint.
    """
    image_paths = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not image_paths:
        print(f"No files matching {pattern} found in {input_dir}")
        return

    fluo_mode = fluo_channel is not None or fluo_index is not None
    if model_p2p_path is None and not fluo_mode:
        raise ValueError("model_p2p_path is required for brightfield input (Pix2Pix BF->fluorescence translation); "
                         "pass fluo_channel or fluo_index to segment a fluorescence channel instead")
    if model_smp_path is None:
        raise ValueError("model_smp_path is required for segmentation")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_p2p = load_pix2pix(model_p2p_path).to('cpu') if not fluo_mode else None
    model_smp = load_segmentation_model(model_smp_path).to('cpu')
    print('Models loaded')

    ref_voxel_size = [5, 2, 2]
    z_anisotropy = ref_voxel_size[0] / np.mean([ref_voxel_size[1], ref_voxel_size[2]])

    for image_path in image_paths:
        try:
            stacks = _iter_stacks(image_path, fluo_channel, fluo_index)
            for filename, (loader, loader_kwargs), channel_names in stacks:
                try:
                    if os.path.exists(os.path.join(input_dir, filename + '_vessel_mask.tif')):
                        print(f'{filename} already processed. Skipped')
                        continue

                    print(f'Analyzing {filename} ...')
                    stack, voxel_size, [nz, ny, nx] = loader(ref_voxel_size=ref_voxel_size, **loader_kwargs)

                    if fluo_mode:
                        current_bf_index = 0                     # single-channel stack
                    elif bf_index is not None:
                        current_bf_index = bf_index
                    else:
                        found = _find_channel(channel_names, ['BF', 'Gray'])
                        current_bf_index = found if found is not None else 0

                    if crop:
                        try:
                            stack = crop_stack(stack, crop_profile, current_bf_index, mode='3D', z_range=z_range,
                                               ref_voxel_size=ref_voxel_size, lif_path_root=input_dir,
                                               filename=filename, profile_dir=crop_profile_dir)
                        except Exception as e:
                            print(f"Cropping failed for {filename}: {e}. Skipped")
                            continue

                    stack_bf = scale(stack[..., current_bf_index])
                    vessel_bf_iso = resize_dask(stack_bf, [z_anisotropy, 1, 1])
                    vessel_fluo_iso = None

                    if fluo_mode:
                        vessel_pred = stack_bf                   # the fluorescence channel itself
                        vessel_fluo_iso = vessel_bf_iso
                        vessel_pred_iso = np.copy(vessel_fluo_iso)
                    else:
                        if stack.shape[-1] > 1:
                            # second channel, if any, shown as the fluorescence reference
                            other_idx = 1 if current_bf_index == 0 else 0
                            vessel_fluo_iso = resize_dask(scale(stack[..., other_idx]), [z_anisotropy, 1, 1])
                        vessel_pred = predict_pix2pix(model_p2p, stack_bf, device, n_iter=1)
                        vessel_pred_iso = resize_dask(vessel_pred, [z_anisotropy, 1, 1])

                    if ortho:
                        vessel_proba_iso = predict_mask_ortho(model_smp, vessel_pred_iso, device)
                        vessel_mask_iso = process_vessel_mask(vessel_proba_iso, ortho)
                    else:
                        vessel_pred_clip = scale(np.clip(vessel_pred, 0.05, 1))
                        vessel_proba = predict_mask(model_smp, vessel_pred_clip, device)
                        vessel_mask = process_vessel_mask(vessel_proba, ortho).astype(np.float32)

                        if save_all_stacks:
                            vessel_pred_iso = resize_dask(vessel_pred_clip, [z_anisotropy, 1, 1])
                        vessel_proba_iso = resize_dask(vessel_proba, [z_anisotropy, 1, 1])
                        vessel_mask_iso = resize_dask(vessel_mask, [z_anisotropy, 1, 1])

                    # Overview: input | probability | mask, as z-sum projections
                    plt.figure(figsize=(30, 15))
                    ax = plt.subplot(131)
                    if vessel_fluo_iso is not None:
                        plt.imshow(contrast(np.sum(vessel_fluo_iso, axis=0), 0.1, 99.9), cmap='magma')
                    else:
                        plt.imshow(contrast(np.sum(vessel_bf_iso, axis=0), 0.1, 99.9), cmap='gray')
                    plt.subplot(132, sharex=ax, sharey=ax)
                    plt.imshow(np.sum(vessel_proba_iso, axis=0), cmap='magma')
                    plt.subplot(133, sharex=ax, sharey=ax)
                    plt.imshow(np.sum(vessel_mask_iso, axis=0), cmap='magma')
                    plt.tight_layout()
                    plt.suptitle(filename, fontsize=40)
                    plt.savefig(os.path.join(input_dir, filename + '_overview.png'), dpi=200)
                    plt.close('all')

                    if save_all_stacks:
                        if vessel_fluo_iso is not None:
                            tif.imwrite(os.path.join(input_dir, filename + '_vessel_fluo.tif'), vessel_fluo_iso.astype(np.float32))
                        tif.imwrite(os.path.join(input_dir, filename + '_vessel_bf.tif'), vessel_bf_iso.astype(np.float32))
                        tif.imwrite(os.path.join(input_dir, filename + '_vessel_pred.tif'), vessel_pred_iso.astype(np.float32))
                        tif.imwrite(os.path.join(input_dir, filename + '_vessel_proba.tif'), vessel_proba_iso.astype(np.float32))

                    tif.imwrite(os.path.join(input_dir, filename + '_vessel_mask.tif'), vessel_mask_iso.astype(np.float32))

                except Exception as e:
                    print(f"Error processing {filename} in {image_path}: {e}")
                    continue
        except Exception as e:
            print(f"Error reading file {image_path}: {e}")

        print('======================================================')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run the VascuMap3D segmentation pipeline.")

    parser.add_argument("--input_dir", "--lif_dir_path", dest="input_dir", type=str, required=True,
                        help="Directory containing the microscopy files (.lif, or any aicsimageio format with --fluo_channel).")
    parser.add_argument("--pattern", type=str, default="*.lif",
                        help="Glob pattern of the files to process, e.g. '*.nd2'.")
    parser.add_argument("--model_p2p_path", type=str, default=None,
                        help="Path to the Pix2Pix checkpoint (brightfield input).")
    parser.add_argument("--model_smp_path", type=str, required=True,
                        help="Path to the segmentation checkpoint.")
    parser.add_argument("--fluo_channel", type=str, default=None,
                        help="Name of a fluorescence vessel channel (e.g. GFP); skips Pix2Pix.")
    parser.add_argument("--fluo_index", type=int, default=None,
                        help="Index of a fluorescence vessel channel; skips Pix2Pix.")
    parser.add_argument("--bf_index", type=int, default=None,
                        help="Index of the brightfield channel (default: found by name).")
    parser.add_argument("--no_crop", action="store_true",
                        help="Skip auto-focus and template cropping.")
    parser.add_argument("--crop_profile", type=str, default="dorota",
                        help="Name of the crop profile.")
    parser.add_argument("--crop_profile_dir", type=str, default="crop_profiles",
                        help="Folder holding the crop profiles.")
    parser.add_argument("--z_range", type=int, default=200,
                        help="Z-range (um) kept around the focal surface when cropping.")
    parser.add_argument("--no_ortho", action="store_true",
                        help="Axial-only segmentation instead of 2.5D averaging.")
    parser.add_argument("--save_all_stacks", action="store_true",
                        help="Also save the intermediate stacks.")

    args = parser.parse_args()

    run(input_dir=args.input_dir, pattern=args.pattern, ortho=not args.no_ortho, bf_index=args.bf_index,
        fluo_channel=args.fluo_channel, fluo_index=args.fluo_index, crop_profile=args.crop_profile,
        crop_profile_dir=args.crop_profile_dir, z_range=args.z_range, save_all_stacks=args.save_all_stacks,
        crop=not args.no_crop, model_p2p_path=args.model_p2p_path, model_smp_path=args.model_smp_path)
