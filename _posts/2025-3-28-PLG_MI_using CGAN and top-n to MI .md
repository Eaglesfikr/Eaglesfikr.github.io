---
title: PLG-MI-using CGAN and top-n to MI
description: This is a novel model inference attack method, We utilized the conditional generation of adversarial networks andtop-n selection strategy labels to guide the training process. Through this method, the search space in the image reconstruction stage can be limited to the subspace of the target category, thereby avoiding interference from other irrelevant features.
author: eaglesfikr
date:   2025-3-28 11:33:00 +0800
categories: [AI security]
tags: [MIA, GAN]
pin: true
math: true
mermaid: true
image:
  path: /assets/commons/image-20250328205351307.png
  lqip: data:image/webp;base64,UklGRpoAAABXRUJQVlA4WAoAAAAQAAAADwAABwAAQUxQSDIAAAARL0AmbZurmr57yyIiqE8oiG0bejIYEQTgqiDA9vqnsUSI6H+oAERp2HZ65qP/VIAWAFZQOCBCAAAA8AEAnQEqEAAIAAVAfCWkAALp8sF8rgRgAP7o9FDvMCkMde9PK7euH5M1m6VWoDXf2FkP3BqV0ZYbO6NA/VFIAAAA
  alt: Responsive rendering of Chirpy theme on multiple devices.
---

There are three types of model inversion attacks (MIA), namely query based model inversion, member inference attacks, and GAN based model inversion attacks. For the third type, GMI was first proposed by Zhang et al. in 2020. Later, Chen et al. proposed that KED-MI utilizes access to the target model to obtain soft labels for guided attacks, while PIG-MI here is modified based on the shortcomings of KED-MI and uses better conditional generation models to guide the generation of images to better reproduce the effects of the private training set.



## background:

The optimization problem of reverse attack based on GAN model can be expressed as follows:

$$
z^* = \arg\min \mathcal{L}_{inv}(T(G(\hat{z})), c)
$$


Among them,c is the true label of the private dataset

The existing GAN based models undergo two search processes in reverse engineering: 

1. searching for generator parameters, and then training on auxiliary datasets (public datasets)

2. Recreate the latent space search of the generator until it approaches the private image

The existing problem with KED-MI is that although it uses soft label guidance (which can be understood as the labels obtained from the image input target model), the generator generated in the first stage is category coupled. When reconstructing the target of a specified category in the second stage, it is necessary to search in the latent space of all categories, which can easily lead to confusion of feature information from different categories. In addition, the semi supervised framework GAN it adopts also relies too much on discriminators to constrain the search space of the generator, lacking specificity for categories.

![image-20250328205351307](https://raw.githubusercontent.com/Eaglesfikr/Eaglesfikr.github.io/main/_posts/img/image-20250328205351307.png)

Since a conditional generator is used, the search only needs to be conducted in the subspace of the specified category. In addition, it adds a random data augmentation module that performs random transformations on the generated images, including adjusting size, cropping, horizontal flipping, rotation, and color jitter. This module provides more stable convergence for generating realistic images while imposing constraints. In order to ensure that the reconstructed image is not deceptive (such as adversarial samples) or only trapped in local minima, it will perform random transformations on the generated image to obtain multiple related views. Intuitively speaking, if the reconstructed image does reveal the key discriminative features of the target category, then its category should remain consistent in these views.So, the formula obtained became like this:

$$
z^* = \arg\min_{\hat{z}} \sum_{i=1}^{m} \mathcal{L}_{inv}(T(\mathcal{A}_i(G(\hat{z}, c))), c)
$$

Among them, A is a set of random data augmentation methods



## Process:

#### 1.Top-n selection strategy:

In order to make public data have pseudo labels to guide the training of the generator, we propose a top-n selection strategy, as shown in Figure 2. This strategy aims to select the best matching n images for each pseudo label from public data. These pseudo labels correspond to classes in the private dataset. Specifically, we input all images from public data into the target model and obtain corresponding prediction vectors. Then, for a certain class k, we sort all images in descending order of the k values in the predicted vector and select the top n images to assign pseudo labels k. After reclassifying public data, we can directly understand the feature distribution of each category image. When reconstructing images of the kth category in a private dataset, simply search for the desired features in Fpub. At the same time, reduce the interference of irrelevant features from the incoming Fpub.

![image-20250328210701066](https://raw.githubusercontent.com/Eaglesfikr/Eaglesfikr.github.io/main/_posts/img/image-20250328210701066.png)

Taking facial recognition as an example, assuming that the kth category in the private dataset is young white male with black hair, then the kth categories in the pseudo labeled public dataset are mostly white people, as shown in Figure 2. Therefore, key features are preserved while useless features (such as other skins or hair) are removed, thereby reducing the search space.



#### 2.Train a CGAN on public data

![image-20250328211621602](https://raw.githubusercontent.com/Eaglesfikr/Eaglesfikr.github.io/main/_posts/img/image-20250328211621602.png)

In order to make the generated image more accurate, we use pseudo labels to impose clear constraints on the generator. In principle, this constraint guides the generated images to belong to a certain category in the private dataset. In addition, we have added a random data augmentation module that performs random transformations on the generated images, including adjusting size, cropping, horizontal flipping, rotation, and color jitter. This module provides more stable convergence for generating realistic images while imposing constraints.In order to make the generated image more accurate, we use pseudo labels to impose clear constraints on the generator. In principle, this constraint guides the generated images to belong to a certain category in the private dataset. In addition, we have added a random data augmentation module that performs random transformations on the generated images, including adjusting size, cropping, horizontal flipping, rotation, and color jitter. This module provides more stable convergence for generating realistic images while imposing constraints.



#### 3.Reconstruct images using a trained generator

![image-20250328211900419](https://raw.githubusercontent.com/Eaglesfikr/Eaglesfikr.github.io/main/_posts/img/image-20250328211900419.png)

After training an adversarial network (GAN) using public data, we can use it to reconstruct images of specified categories in a private dataset, as shown in the figure. Specifically, given the target category c, our goal is to find suitable latent vectors that continuously bring the generated image closer to the images in category c. Since we are using a conditional generator, we only need to search in the subspace of the specified category. To ensure that the reconstructed image is not deceptive (such as adversarial samples) or only captures local minima, we will perform random transformations on the generated image to obtain multiple relevant views. Intuitively speaking, if the reconstructed image does reveal the key discriminative features of the target category, then its category should remain consistent in these views.





## Results:

Using the public dataset ffhq, the target model VGG16 (trained on the private training set celeba)

```
PS D:\workshop\PLG-MI-Attack> python top_n_selection.py --model=VGG16 --data_name=ffhq --top_n=30 --save_root=reclassified_public_data
Namespace(model='VGG16', data_name='ffhq', top_n=30, num_classes=1000, save_root='reclassified_public_data')
=> load target model ...
=> load public dataset ...
Checking path: datasets/ffhq/thumbnails128x128/
len(data_loader) is: 200
len(data_loader): 200
Batch {i}: {type(batch)}
Batch content: {batch}
=> start inference ...
=> start reclassify ...
 top_n:  30
 save_path:  reclassified_public_data\ffhq\VGG16_top30
------------------------------------------------------------------------------------------------------------
root@autodl-container-6fa24bbc0c-707a3ec4:~/REO/PLG-MI-Attack# ulimit -n 65536  
root@autodl-container-6fa24bbc0c-707a3ec4:~/REO/PLG-MI-Attack# python train_cgan.py --data_name=ffhq --target_model=VGG16 --calc_FID --inv_loss_type=margin --max_iteration=30000 --alpha=0.2 --private_data_root=./datasets/celeba_private_domain --data_root=./reclassified_public_data/ffhq/VGG16_top30 --results_root=PLG_MI_Results
Target Model: VGG16
 prepared datasets...
 Number of training images: 30000
{
  "data_root": "./reclassified_public_data/ffhq/VGG16_top30",
  "data_name": "ffhq",
  "target_model": "VGG16",
  "private_data_root": "./datasets/celeba_private_domain",
  "batch_size": 64,
  "eval_batch_size": 16,
  "gen_num_features": 64,
  "gen_dim_z": 128,
  "gen_bottom_width": 4,
  "gen_distribution": "normal",
  "dis_num_features": 64,
  "lr": 0.0002,
  "beta1": 0.0,
  "beta2": 0.9,
  "seed": 46,
  "max_iteration": 30000,
  "n_dis": 5,
  "num_classes": 1000,
  "loss_type": "hinge",
  "relativistic_loss": false,
  "calc_FID": true,
  "results_root": "PLG_MI_Results/ffhq/VGG16",
  "no_tensorboard": false,
  "no_image": false,
  "checkpoint_interval": 1000,
  "log_interval": 100,
  "eval_interval": 1000,
  "n_eval_batches": 1000,
  "n_fid_images": 3000,
  "args_path": null,
  "gen_ckpt_path": null,
  "dis_ckpt_path": null,
  "alpha": 0.2,
  "inv_loss_type": "margin",
  "train_image_root": "PLG_MI_Results/ffhq/VGG16/preview/train",
  "eval_image_root": "PLG_MI_Results/ffhq/VGG16/preview/eval",
  "n_fid_batches": 46
}
Downloading: "https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth" to /root/.cache/torch/hub/checkpoints/inception_v3_google-0cc3c7bd.pth
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 104M/104M [07:00<00:00, 259kB/s]
 Initialized models...

/root/miniconda3/lib/python3.8/site-packages/torch/nn/functional.py:3631: UserWarning: Default upsampling behavior when mode=bilinear is changed to align_corners=False since 0.4.0. Please specify align_corners=True if the old behavior is desired. See the documentation of nn.Upsample for details.
  warnings.warn(
iteration: 0000100/0030000, loss gen: 0.033076, loss dis 1.125432, inv loss 6.508729, target acc 0.002440
iteration: 0000200/0030000, loss gen: 1.299898, loss dis 0.634624, inv loss 5.617334, target acc 0.002040
iteration: 0000300/0030000, loss gen: 1.457960, loss dis 0.627980, inv loss 5.636545, target acc 0.001400
iteration: 0000400/0030000, loss gen: 1.453651, loss dis 0.780714, inv loss 6.448255, target acc 0.001400
iteration: 0000500/0030000, loss gen: 1.608786, loss dis 0.421779, inv loss 6.813730, target acc 0.000000
iteration: 0000600/0030000, loss gen: 1.066864, loss dis 0.924695, inv loss 7.080348, target acc 0.000000
iteration: 0000700/0030000, loss gen: 0.209315, loss dis 0.779688, inv loss 7.151425, target acc 0.009580
iteration: 0000800/0030000, loss gen: 1.442775, loss dis 1.184831, inv loss 8.150633, target acc 0.000000
iteration: 0000900/0030000, loss gen: 0.496013, loss dis 0.895768, inv loss 7.690419, target acc 0.000000
iteration: 0001000/0030000, loss gen: 0.612300, loss dis 0.842832, inv loss 7.740147, target acc 0.000000
[Eval] iteration: 0001000/0030000, FID: 120.019351
iteration: 0001100/0030000, loss gen: 1.011073, loss dis 0.968511, inv loss 8.229614, target acc 0.002440
iteration: 0001200/0030000, loss gen: 1.675528, loss dis 0.993399, inv loss 8.794401, target acc 0.000000
iteration: 0001300/0030000, loss gen: 0.212413, loss dis 1.083812, inv loss 8.721048, target acc 0.000000
iteration: 0001400/0030000, loss gen: 0.413593, loss dis 1.107696, inv loss 7.967240, target acc 0.000000
iteration: 0001500/0030000, loss gen: 1.032235, loss dis 1.139786, inv loss 8.570403, target acc 0.000000
iteration: 0001600/0030000, loss gen: 1.648639, loss dis 1.007083, inv loss 8.248553, target acc 0.000000
iteration: 0001700/0030000, loss gen: 2.185567, loss dis 1.097644, inv loss 7.978811, target acc 0.004000
iteration: 0001800/0030000, loss gen: 1.621488, loss dis 0.966941, inv loss 8.200933, target acc 0.000000
iteration: 0001900/0030000, loss gen: 1.950765, loss dis 0.750152, inv loss 8.370026, target acc 0.001400
iteration: 0002000/0030000, loss gen: 1.799993, loss dis 0.903950, inv loss 8.069005, target acc 0.000620
[Eval] iteration: 0002000/0030000, FID: 36.219149
iteration: 0002100/0030000, loss gen: -0.245668, loss dis 0.977957, inv loss 8.635996, target acc 0.000000
iteration: 0002200/0030000, loss gen: 1.553149, loss dis 0.824746, inv loss 8.542104, target acc 0.000620
iteration: 0002300/0030000, loss gen: 0.493726, loss dis 1.149137, inv loss 8.918242, target acc 0.002440
iteration: 0002400/0030000, loss gen: 0.999577, loss dis 1.198100, inv loss 7.951801, target acc 0.001400
iteration: 0002500/0030000, loss gen: 1.659570, loss dis 0.723352, inv loss 8.099534, target acc 0.000000
iteration: 0002600/0030000, loss gen: 1.736059, loss dis 0.962312, inv loss 7.758273, target acc 0.014260
iteration: 0002700/0030000, loss gen: 1.803399, loss dis 0.910575, inv loss 8.221584, target acc 0.004000
iteration: 0002800/0030000, loss gen: 0.844269, loss dis 1.172973, inv loss 8.640617, target acc 0.000000
iteration: 0002900/0030000, loss gen: 0.475111, loss dis 0.931342, inv loss 7.758621, target acc 0.001260
iteration: 0003000/0030000, loss gen: 1.643094, loss dis 0.771260, inv loss 7.583474, target acc 0.000000
[Eval] iteration: 0003000/0030000, FID: 29.946464
iteration: 0003100/0030000, loss gen: 1.645789, loss dis 0.764987, inv loss 7.753283, target acc 0.000000
iteration: 0003200/0030000, loss gen: 0.552951, loss dis 0.905714, inv loss 7.222503, target acc 0.011140
iteration: 0003300/0030000, loss gen: 1.565990, loss dis 0.842027, inv loss 7.093271, target acc 0.002440
iteration: 0003400/0030000, loss gen: 0.627242, loss dis 1.045482, inv loss 7.857070, target acc 0.014880
iteration: 0003500/0030000, loss gen: 0.639955, loss dis 0.765746, inv loss 7.201542, target acc 0.014880
iteration: 0003600/0030000, loss gen: 1.792219, loss dis 0.742979, inv loss 6.581047, target acc 0.000620
iteration: 0003700/0030000, loss gen: 0.286109, loss dis 0.865889, inv loss 7.234929, target acc 0.009580
iteration: 0003800/0030000, loss gen: 1.532520, loss dis 0.883631, inv loss 6.700103, target acc 0.018980
iteration: 0003900/0030000, loss gen: 2.024024, loss dis 0.884475, inv loss 6.829861, target acc 0.016220
iteration: 0004000/0030000, loss gen: 1.692592, loss dis 0.737353, inv loss 7.142865, target acc 0.007760
[Eval] iteration: 0004000/0030000, FID: 25.902636
iteration: 0004100/0030000, loss gen: 1.032123, loss dis 0.936756, inv loss 5.901046, target acc 0.023620
iteration: 0004200/0030000, loss gen: 1.328900, loss dis 0.727853, inv loss 6.743250, target acc 0.016340
iteration: 0004300/0030000, loss gen: 0.729004, loss dis 0.825139, inv loss 6.140398, target acc 0.007860
iteration: 0004400/0030000, loss gen: 0.332382, loss dis 0.975672, inv loss 5.244941, target acc 0.031660
iteration: 0004500/0030000, loss gen: 1.434266, loss dis 0.935201, inv loss 6.120275, target acc 0.031820
iteration: 0004600/0030000, loss gen: 2.055647, loss dis 0.723180, inv loss 5.989646, target acc 0.029720
iteration: 0004700/0030000, loss gen: 1.858795, loss dis 0.739784, inv loss 4.687065, target acc 0.042820
iteration: 0004800/0030000, loss gen: 1.359205, loss dis 0.653650, inv loss 5.406180, target acc 0.037980
iteration: 0004900/0030000, loss gen: 1.883476, loss dis 0.619927, inv loss 5.436260, target acc 0.036320
iteration: 0005000/0030000, loss gen: 1.426290, loss dis 0.611787, inv loss 5.389750, target acc 0.046200
[Eval] iteration: 0005000/0030000, FID: 20.047341
iteration: 0005100/0030000, loss gen: 1.726979, loss dis 0.716882, inv loss 5.403959, target acc 0.067960
iteration: 0005200/0030000, loss gen: 1.866477, loss dis 0.732646, inv loss 5.048716, target acc 0.054860
iteration: 0005300/0030000, loss gen: 1.692710, loss dis 0.743533, inv loss 5.313179, target acc 0.054020
iteration: 0005400/0030000, loss gen: 0.433937, loss dis 0.976146, inv loss 4.429922, target acc 0.091100
iteration: 0005500/0030000, loss gen: 1.626268, loss dis 0.837507, inv loss 4.120919, target acc 0.071560
iteration: 0005600/0030000, loss gen: 0.539392, loss dis 0.888513, inv loss 3.714131, target acc 0.087600
iteration: 0005700/0030000, loss gen: 1.566703, loss dis 0.756467, inv loss 4.195097, target acc 0.101060
iteration: 0005800/0030000, loss gen: 0.877301, loss dis 0.977488, inv loss 3.426380, target acc 0.124420
iteration: 0005900/0030000, loss gen: 1.592687, loss dis 0.901587, inv loss 2.915566, target acc 0.059860
iteration: 0006000/0030000, loss gen: 0.803164, loss dis 0.883445, inv loss 3.186957, target acc 0.157280
[Eval] iteration: 0006000/0030000, FID: 19.290297
iteration: 0006100/0030000, loss gen: 0.926718, loss dis 0.958812, inv loss 3.152556, target acc 0.169980
iteration: 0006200/0030000, loss gen: 1.555837, loss dis 0.649585, inv loss 3.090879, target acc 0.190020
iteration: 0006300/0030000, loss gen: 1.557656, loss dis 0.591682, inv loss 3.489059, target acc 0.175060
iteration: 0006400/0030000, loss gen: 1.310754, loss dis 0.936729, inv loss 2.467109, target acc 0.224880
iteration: 0006500/0030000, loss gen: 1.305196, loss dis 0.743005, inv loss 1.499222, target acc 0.203860
iteration: 0006600/0030000, loss gen: 2.218958, loss dis 0.820057, inv loss 2.467579, target acc 0.185620
iteration: 0006700/0030000, loss gen: 1.882391, loss dis 0.998350, inv loss 2.246366, target acc 0.205440
iteration: 0006800/0030000, loss gen: 1.190475, loss dis 0.772298, inv loss 2.112936, target acc 0.214060
iteration: 0006900/0030000, loss gen: 1.446115, loss dis 0.697176, inv loss 1.897575, target acc 0.278860
iteration: 0007000/0030000, loss gen: 0.851414, loss dis 0.884666, inv loss 1.761947, target acc 0.263980
[Eval] iteration: 0007000/0030000, FID: 28.476521
iteration: 0007100/0030000, loss gen: 1.574633, loss dis 0.715455, inv loss 1.466202, target acc 0.316300
iteration: 0007200/0030000, loss gen: 1.499983, loss dis 0.645947, inv loss 1.314610, target acc 0.298400
iteration: 0007300/0030000, loss gen: 1.779203, loss dis 0.685084, inv loss 1.630424, target acc 0.391380
iteration: 0007400/0030000, loss gen: 1.959484, loss dis 0.680462, inv loss 1.298815, target acc 0.360100
iteration: 0007500/0030000, loss gen: 1.634617, loss dis 0.712690, inv loss 1.365194, target acc 0.335460
iteration: 0007600/0030000, loss gen: 1.865360, loss dis 0.693320, inv loss 0.242401, target acc 0.390880
iteration: 0007700/0030000, loss gen: 1.685281, loss dis 0.645405, inv loss 0.489553, target acc 0.362800
iteration: 0007800/0030000, loss gen: 1.464802, loss dis 0.822924, inv loss 1.641825, target acc 0.400760
iteration: 0007900/0030000, loss gen: 1.656809, loss dis 0.711343, inv loss 0.018343, target acc 0.401140
iteration: 0008000/0030000, loss gen: 1.653112, loss dis 0.640878, inv loss 0.815213, target acc 0.489640
[Eval] iteration: 0008000/0030000, FID: 18.565738
iteration: 0008100/0030000, loss gen: 1.730250, loss dis 0.727211, inv loss -0.004709, target acc 0.433740
iteration: 0008200/0030000, loss gen: 1.033345, loss dis 0.712511, inv loss -0.174944, target acc 0.540440
iteration: 0008300/0030000, loss gen: 1.693148, loss dis 0.632752, inv loss 1.256298, target acc 0.514720
iteration: 0008400/0030000, loss gen: 1.550017, loss dis 0.782910, inv loss -0.283688, target acc 0.498300
iteration: 0008500/0030000, loss gen: 1.731839, loss dis 0.677775, inv loss 0.176477, target acc 0.584580
iteration: 0008600/0030000, loss gen: 1.092074, loss dis 0.849255, inv loss -0.014256, target acc 0.613220
iteration: 0008700/0030000, loss gen: 1.618413, loss dis 0.659393, inv loss -0.068498, target acc 0.645660
iteration: 0008800/0030000, loss gen: 1.784856, loss dis 0.693628, inv loss -0.462620, target acc 0.585160
iteration: 0008900/0030000, loss gen: 1.413784, loss dis 0.659918, inv loss -0.671216, target acc 0.700980
iteration: 0009000/0030000, loss gen: 2.054783, loss dis 0.776239, inv loss -0.527787, target acc 0.714880
[Eval] iteration: 0009000/0030000, FID: 17.071943
iteration: 0009100/0030000, loss gen: 1.226096, loss dis 0.605419, inv loss -0.921700, target acc 0.655860
iteration: 0009200/0030000, loss gen: 1.870528, loss dis 0.620510, inv loss -0.748536, target acc 0.680820
iteration: 0009300/0030000, loss gen: 1.699623, loss dis 0.651682, inv loss -1.083428, target acc 0.685580
iteration: 0009400/0030000, loss gen: 1.631341, loss dis 0.646831, inv loss -1.055737, target acc 0.688160
iteration: 0009500/0030000, loss gen: 0.701277, loss dis 0.800941, inv loss -1.749521, target acc 0.773000
iteration: 0009600/0030000, loss gen: 1.529408, loss dis 0.637795, inv loss -1.328152, target acc 0.749940
iteration: 0009700/0030000, loss gen: 1.574852, loss dis 0.610442, inv loss -1.290256, target acc 0.709640
...
iteration: 0027300/0030000, loss gen: 1.533352, loss dis 0.399945, inv loss -7.667150, target acc 1.000000
iteration: 0027400/0030000, loss gen: 1.699495, loss dis 0.367870, inv loss -7.102077, target acc 0.997560
iteration: 0027500/0030000, loss gen: 1.522183, loss dis 0.417591, inv loss -7.255234, target acc 1.000000
iteration: 0027600/0030000, loss gen: 1.126593, loss dis 0.506623, inv loss -7.370131, target acc 0.996000
iteration: 0027700/0030000, loss gen: 1.006554, loss dis 0.436462, inv loss -7.237269, target acc 0.992000
iteration: 0027800/0030000, loss gen: 1.482473, loss dis 0.412669, inv loss -6.990086, target acc 1.000000
iteration: 0027900/0030000, loss gen: 1.068959, loss dis 0.492183, inv loss -6.874802, target acc 1.000000
iteration: 0028000/0030000, loss gen: 2.232770, loss dis 0.459355, inv loss -7.570553, target acc 0.990440
[Eval] iteration: 0028000/0030000, FID: 15.834341
iteration: 0028100/0030000, loss gen: 1.484194, loss dis 0.456040, inv loss -7.003890, target acc 1.000000
iteration: 0028200/0030000, loss gen: 0.883972, loss dis 0.559243, inv loss -7.473935, target acc 1.000000
iteration: 0028300/0030000, loss gen: 0.994162, loss dis 0.497140, inv loss -7.390285, target acc 1.000000
iteration: 0028400/0030000, loss gen: 1.160646, loss dis 0.445654, inv loss -7.521864, target acc 0.998600
iteration: 0028500/0030000, loss gen: 1.676934, loss dis 0.361111, inv loss -7.294280, target acc 1.000000
iteration: 0028600/0030000, loss gen: 2.100122, loss dis 0.365549, inv loss -7.307921, target acc 1.000000
iteration: 0028700/0030000, loss gen: 2.046581, loss dis 0.358888, inv loss -7.159393, target acc 1.000000
iteration: 0028800/0030000, loss gen: 1.271441, loss dis 0.480838, inv loss -7.442312, target acc 1.000000
iteration: 0028900/0030000, loss gen: 0.990100, loss dis 0.458739, inv loss -7.246819, target acc 1.000000
iteration: 0029000/0030000, loss gen: 1.935347, loss dis 0.431446, inv loss -7.513340, target acc 1.000000
[Eval] iteration: 0029000/0030000, FID: 17.966263
iteration: 0029100/0030000, loss gen: 1.543362, loss dis 0.441732, inv loss -7.453138, target acc 1.000000
iteration: 0029200/0030000, loss gen: 1.504454, loss dis 0.439809, inv loss -7.424896, target acc 1.000000
iteration: 0029300/0030000, loss gen: 1.082813, loss dis 0.435057, inv loss -7.413267, target acc 1.000000
iteration: 0029400/0030000, loss gen: 1.846408, loss dis 0.363024, inv loss -7.171417, target acc 1.000000
iteration: 0029500/0030000, loss gen: 2.049986, loss dis 0.431750, inv loss -7.483669, target acc 1.000000
iteration: 0029600/0030000, loss gen: 0.726688, loss dis 0.455019, inv loss -7.325363, target acc 1.000000
iteration: 0029700/0030000, loss gen: 1.937887, loss dis 0.365296, inv loss -7.170828, target acc 1.000000
iteration: 0029800/0030000, loss gen: 1.778177, loss dis 0.358284, inv loss -7.690294, target acc 1.000000
iteration: 0029900/0030000, loss gen: 0.990790, loss dis 0.497546, inv loss -6.789579, target acc 1.000000
iteration: 0030000/0030000, loss gen: 2.040687, loss dis 0.379708, inv loss -7.298969, target acc 1.000000
[Eval] iteration: 0030000/0030000, FID: 17.014978
#Performing the "ls" command yields the following result：
```

![image-20250328190505140](https://raw.githubusercontent.com/Eaglesfikr/Eaglesfikr.github.io/main/_posts/img/image-20250328190505140.png)

In order to save time here, we didn't pass all of them down. We only took a few special files and folders(4 col is for one images):`preview/eval/class_id_001`

![joined-image1](https://raw.githubusercontent.com/Eaglesfikr/Eaglesfikr.github.io/main/_posts/img/joined-image1.png)

![joined-image2](https://raw.githubusercontent.com/Eaglesfikr/Eaglesfikr.github.io/main/_posts/img/joined-image2.png)

![joined-image3](https://raw.githubusercontent.com/Eaglesfikr/Eaglesfikr.github.io/main/_posts/img/joined-image3-1743238259824.png)

![joined-image](https://raw.githubusercontent.com/eaglesfikr/eaglesfikr.github.io/main/_posts/img/joined-image.png)

It can be seen that for the second character, private characteristics such as gender, skin color, face shape, and eyes were gradually learned!!!

![joined-image3](https://raw.githubusercontent.com/Eaglesfikr/Eaglesfikr.github.io/main/_posts/img/joined-image4.png)

This is the classification of its private model...

