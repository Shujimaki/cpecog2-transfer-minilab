|                               |   accuracy |   macro_f1 |   trainable_params |   train_time_s |
|:------------------------------|-----------:|-----------:|-------------------:|---------------:|
| Small CNN (scratch)           |      0.690 |      0.689 |             94,341 |        106.393 |
| ResNet-50 feature extraction  |      0.910 |      0.910 |             10,245 |        502.742 |
| ResNet-50 partial fine-tuning |      0.923 |      0.924 |         14,974,981 |        111.747 |
| ResNet-50 full fine-tuning    |      0.930 |      0.930 |         23,518,277 |        270.181 |
