# LLM bio hackathon (Ultrasound image simulator)


## Python environment configuration

Install `conda` on linux with command
```
wget https://repo.anaconda.com/archive/Anaconda3-2023.07-1-Linux-x86_64.sh
```
or the latest version from https://www.anaconda.com/download#downloads for your particular machine.
Then do:
```
bash Anaconda3-2023.07-1-Linux-x86_64.sh
```

Now create a conda enviroment with name `ultrasound`
```
conda env create -f environment.yml
```
Finally, activate the environment by
```
conda activate ultrasound
```


## Data preparation

The data that we use to fine-tune is the [Breast Ultrasound Images Dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset), which contains 780 images with size 500*500.
The images are categorized into three classes, which are normal, benign, and malignant.
One can download the dataset from the link 
https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset.
The images will be stored in the folder `Dataset_BUSI_with_GT`.



## Data processing

After downloading the ultrasound dataset, create a folder `ultrasound` to store all the ultrasound images that exclude the **mask**.
Then create three folders `benign`, `malignant` and `normal` to store the corresponded class of images.


This can be done by runing 
```
python data_loading.py
```

Remember to change the `source_dir` and `destination_dir` for different classes in the `data.py`.
Finally, this will remove the mask images from the classes and copy the images into the `ultrasound` folder.
The dataset and environment are now ready!




## Fine-tuning

The training pipeline is mainly based on https://huggingface.co/docs/diffusers/v0.11.0/en/training/dreambooth.

Specifically, we use the stable-diffusion models for fine-tuning
based on two methods: `DreamBooth`, `DreamBooth with LORA`.
For each method, we fine-tune in two classes of stable-diffusion models:
`stable-diffusion-v1`, `stable-diffusion-2`.

The fine-tuning process follows by the order 
`normal->benign->malignant` with totally 3 ruonds of training.


To run the code, firstly install Diffusers from Github:
```
pip install git+https://github.com/huggingface/diffusers
pip install -U -r diffusers/examples/dreambooth/requirements.txt
```

After all the dependencies have been set up, initialize a Accelerate environment with:
```
accelerate config
```


Then run the following code to fine-tune in a stable-diffusion-v1 model by DreamBooth with LORA method.
```
bash dreambooth_lora_v1.sh
```

For stable-diffusion-v1 model + DreamBooth method, run
```
bash dreambooth_v1.sh
```

For stable-diffusion-2 model + DreamBooth method, run
```
bash dreambooth_2.sh
```

For stable-diffusion-2 model + DreamBooth with LORA method, run
```
bash dreambooth_lora_2.sh
```





## Inference





