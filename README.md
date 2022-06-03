# tf-garf

An unofficial implementation of Gaussian Activated Radiance Fields in TensorFlow 2.

Original paper by Chng et al. (2022): [[arXiv]](https://arxiv.org/abs/2204.05735) [[webpage]](https://sfchng.github.io/garf/)

See the [notebook](/garf.ipynb) for more info.

## Data

This implementation expects LLFF-style, forward-facing data, such as the [NeRF LLFF dataset](https://drive.google.com/drive/folders/14boI-o5hGO9srnWaaogTU5_ji7wkX2S7).

```python
from lib.data import load_data
data = load_data('path/to/data/root')
```

If you have other data, you can use it like so.

```python
from lib.data import hwf_rgb_to_data

hwf = np.array([image_height, image_width, focal_length]) # in pixels
rgb = # ... list of np.array objects of shape (image_height, image_width, 3)

data = hwf_rgb_to_data(hwf, rgb)
```

## Model

Once you have the data, you can create a model object.

```python
from model.garf import GaussianRadianceField
model = GaussianRadianceField(data, num_samples=128)
```

You can load the pre-trained model for the NeRF `flower` dataset. Pass `opt=True` to load the optimizer state.

```python
model.load('pretrain/flowers', opt=False)
````

Render a view from a specific image index or pose with [`model.garf.GaussianRadianceField.render`](https://github.com/laura-a-n-n/tf-garf/blob/main/model/garf.py#L160).
