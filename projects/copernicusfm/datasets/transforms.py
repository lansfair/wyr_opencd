import numpy as np
import mmcv
import mmengine.fileio as fileio
from mmcv.transforms import BaseTransform

from opencd.registry import TRANSFORMS

try:
    from osgeo import gdal
except ImportError:
    gdal = None


@TRANSFORMS.register_module()
class MultiImgLoadGeoTiffImageFromFile(BaseTransform):
    """Load a pair of multi-band GeoTIFF images."""

    def __init__(self,
                 band_indices=None,
                 band_scales=None,
                 nan_to_num=True,
                 to_float32=True):
        if gdal is None:
            raise RuntimeError('gdal is not installed')
        self.band_indices = band_indices
        self.band_scales = None if band_scales is None else np.array(
            band_scales, dtype=np.float32)
        self.nan_to_num = nan_to_num
        self.to_float32 = to_float32

    def _read_one_file(self, filename, apply_band_scales=True):
        ds = gdal.Open(filename)
        if ds is None:
            raise Exception(f'Unable to open file: {filename}')
        if self.band_indices is None:
            img = ds.ReadAsArray()
            if img.ndim == 2:
                img = img[None, ...]
        else:
            bands = []
            for index in self.band_indices:
                band = ds.GetRasterBand(index)
                if band is None:
                    raise ValueError(
                        f'Band index {index} is out of range for {filename}')
                bands.append(band.ReadAsArray())
            img = np.stack(bands)
        if self.to_float32:
            img = img.astype(np.float32)
        if self.nan_to_num:
            img = np.nan_to_num(img)
        if apply_band_scales and self.band_scales is not None:
            if len(self.band_scales) != img.shape[0]:
                raise ValueError('band_scales length must match image bands.')
            img = img * self.band_scales.reshape(-1, 1, 1)
        return np.transpose(img, (1, 2, 0))

    def _read_image(self, filename_or_band_files):
        if isinstance(filename_or_band_files, (list, tuple)):
            bands = []
            for filename in filename_or_band_files:
                img = self._read_one_file(filename, apply_band_scales=False)
                if img.ndim == 3 and img.shape[-1] == 1:
                    img = img[..., 0]
                bands.append(img)
            img = np.stack(bands, axis=-1)
            if self.to_float32:
                img = img.astype(np.float32)
            if self.nan_to_num:
                img = np.nan_to_num(img)
            if self.band_scales is not None:
                if len(self.band_scales) != img.shape[-1]:
                    raise ValueError('band_scales length must match image bands.')
                img = img * self.band_scales.reshape(1, 1, -1)
            return img
        return self._read_one_file(filename_or_band_files)

    def transform(self, results):
        imgs = [self._read_image(filename) for filename in results['img_path']]
        if imgs[0].shape[0:2] != imgs[1].shape[0:2]:
            raise Exception(f'Image shapes do not match: '
                            f'{imgs[0].shape} vs {imgs[1].shape}')
        results['img'] = imgs #OSCD用
        print(imgs[0].shape,imgs[1].shape)
        results['img_shape'] = imgs[0].shape[:2]
        results['ori_shape'] = imgs[0].shape[:2]
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'band_indices={self.band_indices}, '
                    f'band_scales={self.band_scales}, '
                    f'nan_to_num={self.nan_to_num}, '
                    f'to_float32={self.to_float32})')
        return repr_str


@TRANSFORMS.register_module()
class MultiImgNormalizeMultibandImage(BaseTransform):
    """Normalize each image in a multi-band image pair."""

    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, -1)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, -1)

    def transform(self, results):
        results['img'] = [
            (img.astype(np.float32) - self.mean) / self.std
            for img in results['img']
        ]
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(mean={self.mean}, std={self.std})'
