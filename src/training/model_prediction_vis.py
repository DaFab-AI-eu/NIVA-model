import logging
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import tensorflow as tf
from eoflow.models.segmentation_unets import ResUnetA

logging.getLogger().setLevel(logging.ERROR)

def visualization_parcels(data, label, num_plt=5, beta=0.3, alpha=0.3,
                          label_="GT", ind_s=0, num_h=1):
    plt.subplot(num_h, num_plt, ind_s + 1)
    bgr = data['features'][0, :, :, :3].numpy()  # with normalization
    rgb = bgr[..., [2, 1, 0]]
    plt.imshow(rgb)
    plt.title('RGB')
    plt.subplot(num_h, num_plt, ind_s + 2)
    extent = np.argmax(label['extent'], axis=-1)[0]
    plt.imshow(extent, cmap='gray')
    plt.title(f'{label_} extent')
    plt.subplot(num_h, num_plt, ind_s + 3)
    boundary = np.argmax(label['boundary'], axis=-1)[0]
    plt.imshow(boundary, cmap='gray')
    plt.title(f'{label_} boundary')
    blended_image = (1 - alpha) * rgb + alpha * extent[:, :, np.newaxis]
    plt.subplot(num_h, num_plt, ind_s + 4)
    plt.imshow(blended_image)
    plt.title(f'{label_} blend RGB & extent')
    blended_image = (1 - beta) * rgb + beta * boundary[:, :, np.newaxis]
    plt.subplot(num_h, num_plt, ind_s + 5)
    plt.imshow(blended_image)
    plt.title(f'{label_} blend RGB & boundary')


abs_path = "/home/niva/ai4boundaries_data/models/training_20241030_031952" # change to your own
# training_20241030_031952 training_20241029_162851 training_20241029_122232 training_20241029_093604
# training_20241028_141719 training_20241028_195657 training_20241029_040542
chkpt_folder = os.path.join(abs_path, "checkpoints")  # change to your own
model_cfg_path = os.path.join(abs_path, "model_cfg.json")  # change to your own
fold = "test"  # or val
dataset_path = f"/home/niva/ai4boundaries_data/ai4boundaries_data/training_data/datasets/{fold}"  # change to your own
path_to_pdf = os.path.join(abs_path, f'{fold}_vis.pdf')  # change to your own
max_img = 40

if __name__ == "__main__":
    input_shape = (256, 256, 4)
    input_shape = dict(features=[None, *input_shape])
    with open(model_cfg_path, 'r') as jfile:
        model_cfg = json.load(jfile)
    # initialise model from config, build, compile and load trained weights
    model = ResUnetA(model_cfg)
    model.build(input_shape)
    model.net.compile()
    model.net.load_weights(f'{chkpt_folder}/model.ckpt')
    # print(model.net.summary())
    # loading tensorflow dataset
    batch_size = 1
    dataset = tf.data.Dataset.load(dataset_path)
    # print(dataset)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # visualize predictions
    with PdfPages(path_to_pdf) as pdf:  # diff normalization params
        for ind_n, (data, label) in enumerate(dataset):
            pred = model.net.predict(data)
            pred_l = {"distance": pred[0], "boundary": pred[1], "extent": pred[2]}
            fig = plt.figure(figsize=(15, 7))
            visualization_parcels(data, label, label_='GT', num_h=2, ind_s=0)
            visualization_parcels(data, pred_l, label_='pred', num_h=2, ind_s=5)
            plt.show()
            pdf.savefig(fig)
            plt.close()

            if ind_n > max_img:
                break
