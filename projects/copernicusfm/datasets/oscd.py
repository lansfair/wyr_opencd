import glob
import os
import os.path as osp
import re
from typing import List

import mmengine

from opencd.registry import DATASETS
from opencd.datasets.basecddataset import _BaseCDDataset


@DATASETS.register_module()
class OSCD_Dataset(_BaseCDDataset):
    """OSCD binary change detection dataset."""

    METAINFO = dict(
        classes=('unchanged', 'changed'),
        palette=[[0, 0, 0], [255, 255, 255]])
    ALL_BANDS = (
        'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A',
        'B09', 'B10', 'B11', 'B12')

    def __init__(self,
                 img_suffix='.tif',
                 seg_map_suffix='.png',
                 format_seg_map='to_binary',
                 official_format=False,
                 split='train',
                 bands=ALL_BANDS,
                 **kwargs):
        self.official_format = official_format
        self.split = split
        self.bands = tuple(bands)
        if not set(self.bands).issubset(set(self.ALL_BANDS)):
            raise ValueError(f'Unsupported OSCD bands: {self.bands}')
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            format_seg_map=format_seg_map,
            **kwargs)

    def _sort_sentinel2_bands(self, paths):
        order = {band: i for i, band in enumerate(self.ALL_BANDS)}

        def get_band_index(path):
            basename = osp.basename(path)
            match = re.search(r'B(?:0[1-9]|1[0-2]|8A)', basename)
            if match is None:
                raise ValueError(f'Cannot parse Sentinel-2 band from {path}')
            return order[match.group(0)]

        return sorted(paths, key=get_band_index)

    def _get_band_paths(self, images_root, region, temporal_index):
        pattern = osp.join(images_root, region,
                           f'imgs_{temporal_index}_rect', '*.tif')
        paths = self._sort_sentinel2_bands(glob.glob(pattern))
        band_to_path = {}
        for path in paths:
            band = re.search(r'B(?:0[1-9]|1[0-2]|8A)', osp.basename(path))
            if band is not None:
                band_to_path[band.group(0)] = path
        missing = [band for band in self.bands if band not in band_to_path]
        if missing:
            raise FileNotFoundError(
                f'Missing bands for {region} time {temporal_index}: '
                f'{missing}')
        return [band_to_path[band] for band in self.bands]

    def load_data_list(self) -> List[dict]:
        if not self.official_format:
            return super().load_data_list()

        split_name = self.split.capitalize()
        images_root = osp.join(
            self.data_root, 'Onera Satellite Change Detection dataset - Images')
        labels_root = osp.join(
            self.data_root,
            f'Onera Satellite Change Detection dataset - {split_name} Labels')
        if not osp.isdir(images_root):
            raise FileNotFoundError(f'OSCD images folder not found: {images_root}')
        if not osp.isdir(labels_root):
            raise FileNotFoundError(f'OSCD labels folder not found: {labels_root}')

        regions = None
        if self.ann_file and osp.isfile(self.ann_file):
            regions = set(
                line.strip() for line in mmengine.list_from_file(self.ann_file)
                if line.strip())

        data_list = []
        for folder in sorted(glob.glob(osp.join(labels_root, '*'))):
            if not osp.isdir(folder):
                continue
            region = osp.basename(folder)
            if regions is not None and region not in regions:
                continue
            mask = osp.join(labels_root, region, 'cm', f'{region}-cm.tif')
            if not osp.isfile(mask):
                mask = osp.join(labels_root, region, 'cm', 'cm.png')
            if not osp.isfile(mask):
                raise FileNotFoundError(f'OSCD mask not found: {mask}')
            data_info = dict(
                img_path=[
                    self._get_band_paths(images_root, region, 1),
                    self._get_band_paths(images_root, region, 2)
                ],
                seg_map_path=mask,
                label_map=self.label_map,
                format_seg_map=self.format_seg_map,
                reduce_zero_label=self.reduce_zero_label,
                seg_fields=[])
            data_list.append(data_info)
        return data_list

@DATASETS.register_module()
class LEVIRCD_Dataset(_BaseCDDataset):
    """LEVIR-CD binary change detection dataset (matches your folder structure)."""

    METAINFO = dict(
        classes=('unchanged', 'changed'),
        palette=[[0, 0, 0], [255, 255, 255]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 format_seg_map='to_binary',
                 **kwargs):
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            format_seg_map=format_seg_map,
            **kwargs)

    def load_data_list(self) -> List[dict]:
        data_list = []

        # 完全匹配你的文件夹：T1 / T2 / GT
        img1_dir = osp.join(self.data_root, self.split, 'T1')
        img2_dir = osp.join(self.data_root, self.split, 'T2')
        label_dir = osp.join(self.data_root, self.split, 'GT')

        # 自动读取所有同名文件
        for fname in sorted(self.list_dir_or_file(img1_dir, list_dir=False)):
            img1_path = osp.join(img1_dir, fname)
            img2_path = osp.join(img2_dir, fname)
            seg_map_path = osp.join(label_dir, fname)

            data_info = dict(
                img_path=[img1_path, img2_path],
                seg_map_path=seg_map_path,
                label_map=self.label_map,
                format_seg_map=self.format_seg_map,
                reduce_zero_label=self.reduce_zero_label,
                seg_fields=[])
            data_list.append(data_info)

        return data_list
