from datasets import Dataset, DatasetDict, Features, Image, Value
from diffusers import (
    AutoencoderKL, DDPMScheduler, StableDiffusionPipeline,
    UNet2DConditionModel)
from diffusers.optimization import get_scheduler
from diffusers.utils import convert_state_dict_to_diffusers
import kagglehub
import logging
import os
import pandas as pd
from peft import LoraConfig
from peft.tuners.lora.layer import LoraLayer
from peft.utils import get_peft_model_state_dict
from PIL import Image as PILImage
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer


RESOLUTION = 512


def _crop_card_img(img, is_pendulum_monster):
    if is_pendulum_monster:
        crop_vals = (65, 245, 845, 810)
    else:
        crop_vals = (140, 290, 770, 920)
    return img.resize((900, 1314)).crop(crop_vals)


def _make_square_and_resize(image):
    width, height = image.size

    # determine the size of the square
    if width > height:
        # add black bars to top and bottom
        new_size = width
        result = PILImage.new('RGB', (new_size, new_size), (0, 0, 0))
        vertical_offset = (new_size - height) // 2
        result.paste(image, (0, vertical_offset))
    elif height > width:
        # add black bars to left and right
        new_size = height
        result = PILImage.new('RGB', (new_size, new_size), (0, 0, 0))
        horizontal_offset = (new_size - width) // 2
        result.paste(image, (horizontal_offset, 0))
    else:
        # already square
        result = image

    # resize to desired size
    result = result.resize((RESOLUTION, RESOLUTION))
    return result


class CardCrafterImageGenerator:
    def __init__(self, dataset_path='cardcrafter_dataset.csv',
                 lora_dir='./cardcrafter_lora'):
        self.base_model_name = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.lora_dir = lora_dir
        self.dataset_path = dataset_path

        # get logger
        self.logger = logging.getLogger('CardCrafter_Logger')
        self.logger.setLevel(logging.INFO)

    def train(self, rank, alpha, batch_size, num_epochs,
              train_text_encoder=False, using_ampere_gpu=False,
              text_encoder_rank=None, text_encoder_alpha=None):
        self.logger.info(
            'Preparing dataset for Yu-Gi-Oh! card image generation...')

        # load Yu-Gi-Oh! card dataset from Kaggle
        dataset = 'archanghosh/yugioh-database'
        kaggle_dataset_path = kagglehub.dataset_download(dataset)

        # load the processed dataset CSV file
        dataset_df = pd.read_csv(self.dataset_path)

        # only training on trap cards and spell cards for now
        dataset_df = dataset_df[
            dataset_df['COMMON_Card_Type'].isin({'Trap', 'Spell'})].copy(
                deep=True)

        # get images and descriptions
        imgs = []
        img_descs = []
        for _, row in dataset_df.iterrows():
            # get image
            types = row['MONSTER_Types']
            is_pendulum = not pd.isna(types) and 'pendulum' in types.lower()
            img_path = os.path.join(
                kaggle_dataset_path, 'Yugi_images', row['COMMON_Image_Name'])
            img = PILImage.open(img_path)
            img = _make_square_and_resize(_crop_card_img(img, is_pendulum))
            imgs.append(img)

            # get image description
            img_descs.append(row['COMMON_Image_Description'])

        # assemble dataset and column names
        data = {'image': imgs, 'text': img_descs}
        dataset = Dataset.from_dict(
            data, features=Features(
                {'image': Image(), 'text': Value('string')}))
        dataset_dict = DatasetDict({'train': dataset})

        self.logger.info(
            'Prepared dataset for Yu-Gi-Oh! card image generation...')

        '''
        The following code of this method was adapted from these scripts
        in the Hugging Face diffusers repository:
        https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py
        https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora_sdxl.py
        '''

        # get diffusion model's noise scheduler, tokenizer, text encoder, VAE,
        # and U-Net from Hugging Face
        self.logger.info(f'Loading components of {self.base_model_name}...')
        noise_scheduler = DDPMScheduler.from_pretrained(
            self.base_model_name, subfolder='scheduler')
        tokenizer = CLIPTokenizer.from_pretrained(
            self.base_model_name, subfolder='tokenizer')
        text_encoder = CLIPTextModel.from_pretrained(
            self.base_model_name, subfolder='text_encoder')
        vae = AutoencoderKL.from_pretrained(
            self.base_model_name, subfolder='vae')
        unet = UNet2DConditionModel.from_pretrained(
            self.base_model_name, subfolder='unet')
        self.logger.info(f'Loaded {self.base_model_name} components')

        # disable gradient calculation, since it is only required for LoRA
        # weights
        unet.requires_grad_(False)
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)

        # cast to desired data type
        weight_dtype = torch.float32
        unet.to(self.device, dtype=weight_dtype)
        vae.to(self.device, dtype=weight_dtype)
        text_encoder.to(self.device, dtype=weight_dtype)

        # add LoRA weights
        unet_lora_config = LoraConfig(
            r=rank, lora_alpha=alpha,
            init_lora_weights='gaussian',
            target_modules=['to_k', 'to_q', 'to_v', 'to_out.0'],
        )
        unet.add_adapter(unet_lora_config)
        if train_text_encoder:
            if text_encoder_rank is None:
                text_encoder_rank = rank
            if text_encoder_alpha is None:
                text_encoder_alpha = alpha
            text_lora_config = LoraConfig(
                r=text_encoder_rank, lora_alpha=text_encoder_alpha,
                init_lora_weights='gaussian',
                target_modules=['q_proj', 'k_proj', 'v_proj', 'out_proj'])
            text_encoder.add_adapter(text_lora_config)
        self.logger.info('Added LoRA weights')

        # enable TF32 for faster training on Ampere GPUs,
        if torch.cuda.is_available() and using_ampere_gpu:
            torch.backends.cuda.matmul.allow_tf32 = True

        # create optimizer
        lora_params = list(filter(
            lambda p: p.requires_grad, unet.parameters()))
        if train_text_encoder:
            lora_params = (
                lora_params
                + list(filter(
                    lambda p: p.requires_grad, text_encoder.parameters()))
                + list(filter(
                    lambda p: p.requires_grad, text_encoder.parameters()))
            )
        optimizer = torch.optim.AdamW(lora_params)

        # function to tokenize image descriptions
        def _tokenize_descs(examples):
            descs = list(examples['text'])
            inputs = tokenizer(
                descs, max_length=tokenizer.model_max_length,
                padding='max_length', truncation=True, return_tensors='pt')
            return inputs.input_ids

        # data preprocessing transforms
        train_transforms = transforms.Compose([
            transforms.Resize(RESOLUTION),
            transforms.CenterCrop(RESOLUTION),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])])

        # utility function for preprocessing data
        def _preprocess_train(examples):
            images = [image.convert('RGB') for image in examples['image']]
            examples['pixel_values'] = [
                train_transforms(image) for image in images]
            examples['input_ids'] = _tokenize_descs(examples)
            return examples

        # set the training transforms on the dataset
        train_dataset = dataset_dict['train'].with_transform(_preprocess_train)

        # utility function for collating data
        def _collate_fn(examples):
            pixel_values = torch.stack(
                [example['pixel_values'] for example in examples])
            pixel_values = pixel_values.to(
                memory_format=torch.contiguous_format).float()
            input_ids = torch.stack(
                [example['input_ids'] for example in examples])
            return {'pixel_values': pixel_values, 'input_ids': input_ids}

        # create dataloader
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, shuffle=True, collate_fn=_collate_fn,
            batch_size=batch_size)

        # get learning rate scheduler
        num_training_steps = num_epochs * len(train_dataloader)
        lr_scheduler = get_scheduler(
            'cosine', optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps)

        # training
        self.logger.info('Fine-tuning with LoRA...')
        progress_bar = tqdm(range(0, num_training_steps), desc='Steps')
        for epoch in range(num_epochs):
            unet.train()
            if train_text_encoder:
                text_encoder.train()
            for step, batch in enumerate(train_dataloader):
                # get latent space embeddings of images
                latents = vae.encode(batch['pixel_values'].to(
                        self.device, dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # add noise to latent space embeddings
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],), device=latents.device)
                timesteps = timesteps.long()
                noisy_latents = noise_scheduler.add_noise(
                    latents, noise, timesteps)

                # get text embedding
                encoder_hidden_states = text_encoder(
                    batch['input_ids'].to(self.device), return_dict=False)[0]

                # predict the noise residual
                model_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states,
                    return_dict=False)[0]

                # calculate loss
                if noise_scheduler.config.prediction_type == 'epsilon':
                    target = noise
                elif noise_scheduler.config.prediction_type == 'v_prediction':
                    target = noise_scheduler.get_velocity(
                        latents, noise, timesteps)
                else:
                    raise ValueError(
                        'Unknown prediction type '
                        + f'{noise_scheduler.config.prediction_type}')
                loss = F.mse_loss(
                    model_pred.float(), target.float(), reduction='mean')

                # backpropagate
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # update progress bar
                progress_bar.update(1)
        self.logger.info('Fine-tuning with LoRA complete')

        # save LoRA weights
        unet_lora_state_dict = convert_state_dict_to_diffusers(
            get_peft_model_state_dict(unet))
        if train_text_encoder:
            text_encoder_lora_layers = convert_state_dict_to_diffusers(
                get_peft_model_state_dict(text_encoder))
        else:
            text_encoder_lora_layers = None
        StableDiffusionPipeline.save_lora_weights(
            save_directory=self.lora_dir,
            unet_lora_layers=unet_lora_state_dict,
            text_encoder_lora_layers=text_encoder_lora_layers,
            safe_serialization=True)
        self.logger.info(f'Saved LoRA weights to {self.lora_dir}')

        # save LoRA configs
        if train_text_encoder:
            _save_lora_configs(self.lora_dir, unet_lora_config,
                               text_lora_config)
        else:
            _save_lora_configs(self.lora_dir, unet_lora_config)
        self.logger.info(f'Saved LoRA config(s) to {self.lora_dir}')

        # free up memory
        del unet
        del text_encoder
        del text_encoder_lora_layers
        torch.cuda.empty_cache()

    def generate(self, prompt):
        # load pipeline with LoRA weights
        if not hasattr(self, 'pipeline'):
            pipeline = StableDiffusionPipeline.from_pretrained(
                self.base_model_name, torch_dtype=torch.float32)
            self.logger.info(
                f'Loaded Stable Diffusion ({self.base_model_name}) pipeline')
            pipeline.load_lora_weights(self.lora_dir)
            pipeline = pipeline.to(self.device)

            # set LoRA alpha
            _set_pipeline_lora_alpha(pipeline, self.lora_dir)

            self.logger.info(f'Loaded LoRA weights from {self.lora_dir}')

            self.pipeline = pipeline

        # generate image
        self.logger.info('Generating image for Yu-Gi-Oh! card...')
        img = self.pipeline(prompt).images[0]
        self.logger.info('Generated image for Yu-Gi-Oh! card')
        return img


'''
The below functions are adapted from the following GitHub issue
discussion comment:
https://github.com/huggingface/diffusers/issues/6087#issuecomment-1846485514

This code is a fix for a bug in the Hugging Face diffusers library where the
alpha parameter of the LoRA config is not loaded correctly when loading
saved LoRA weights. Link to the GitHub issue:
https://github.com/huggingface/diffusers/issues/6087
'''

ADAPTER_NAME = 'default_0'


def _save_lora_configs(lora_dir, unet_lora_config,
                       text_lora_config=None):
    unet_lora_config.save_pretrained(os.path.join(lora_dir, 'unet'))
    if text_lora_config is not None:
        text_lora_config.save_pretrained(
            os.path.join(lora_dir, 'text_encoder'))


def _set_pipeline_lora_alpha(
        pipeline, lora_dir):
    # set LoRA alpha for U-Net
    unet_lora_config_path = os.path.join(lora_dir, 'unet')
    unet_lora_config = LoraConfig.from_pretrained(unet_lora_config_path)
    _set_model_lora_alpha(pipeline.unet, unet_lora_config.lora_alpha)

    # set LoRA alpha for text encoder
    text_lora_config_path = os.path.join(lora_dir, 'text_encoder')
    if os.path.isdir(text_lora_config_path):
        text_lora_config = LoraConfig.from_pretrained(
            text_lora_config_path)
        _set_model_lora_alpha(
            pipeline.text_encoder, text_lora_config.lora_alpha)


def _set_model_lora_alpha(model, lora_alpha):
    # for each LoRA layer in the model, set the alpha value
    for _, module in model.named_modules():
        if isinstance(module, LoraLayer):
            _set_lora_alpha(module, lora_alpha)


def _set_lora_alpha(lora_layer, lora_alpha):
    adapter = ADAPTER_NAME

    # Modified from peft.tuners.lora.layer.LoraLayer.
    if adapter not in lora_layer.active_adapters:
        return

    # if the LoRA layer is active, then set the alpha and scaling
    lora_layer.lora_alpha[adapter] = lora_alpha
    if lora_layer.r[adapter] > 0:
        lora_layer.scaling[adapter] = (
            lora_layer.lora_alpha[adapter] / lora_layer.r[adapter])
