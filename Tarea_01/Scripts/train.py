from Modules.dataset import dataset_model
from Modules.params import get_params
from Modules.pix2pix import pix2pix

params = get_params()
dataset = dataset_model(params)
model = pix2pix()
model.fit(dataset.train,
          4000)
