from cardcrafter_img_gen import CardCrafterImageGenerator
import logging
from math import floor, ceil
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import textwrap
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig


SPELL_TRAP_TEMPLATE_DICT = {
    'templates_filepath': './card_templates/spell_and_trap_templates.jpg',
    'rows': 3,
    'cols': 4
}

CARD_TEMPLATE_MAPPINGS = {
    'NORMAL_SPELL': {
        **SPELL_TRAP_TEMPLATE_DICT,
        'index': 0
    },
    'EQUIP_SPELL': {
        **SPELL_TRAP_TEMPLATE_DICT,
        'index': 1
    },
    'QUICK_PLAY_SPELL': {
        **SPELL_TRAP_TEMPLATE_DICT,
        'index': 2
    },
    'CONTINUOUS_SPELL': {
        **SPELL_TRAP_TEMPLATE_DICT,
        'index': 3
    },
    'RITUAL_SPELL': {
        **SPELL_TRAP_TEMPLATE_DICT,
        'index': 5
    },
    'FIELD_SPELL': {
        **SPELL_TRAP_TEMPLATE_DICT,
        'index': 6
    },
    'NORMAL_TRAP': {
        **SPELL_TRAP_TEMPLATE_DICT,
        'index': 7
    },
    'CONTINUOUS_TRAP': {
        **SPELL_TRAP_TEMPLATE_DICT,
        'index': 8
    },
    'COUNTER_TRAP': {
        **SPELL_TRAP_TEMPLATE_DICT,
        'index': 9
    }
}


def _create_prompt(name, effect):
    prompt = 'Describe an image that would be in the middle of a '\
        + 'Yu-Gi-Oh! card with the following details. Do not answer '\
        + 'in a complete sentence. Use at most 50 words.\n'
    prompt += f'NAME: {name}\n'
    prompt += f'EFFECT: {effect}\n'
    return prompt


def _get_card_template(card_type):
    # check card type
    valid_card_types = set(CARD_TEMPLATE_MAPPINGS.keys())
    if card_type not in valid_card_types:
        raise Exception(f'Invalid card type: {card_type}. Expected one of: '
                        + f'{valid_card_types}')

    # get template info
    template_dict = CARD_TEMPLATE_MAPPINGS[card_type]
    image_path = template_dict['templates_filepath']
    rows = template_dict['rows']
    cols = template_dict['cols']
    index = template_dict['index']

    # load image of card templates
    image = Image.open(image_path)
    width, height = image.size

    # get dimensions of each card template
    part_width = width / cols
    part_height = height / rows

    # calculate row and column of part from index
    row = index // cols
    col = index % cols

    # crop and return the part
    left = floor(col * part_width)
    upper = floor(row * part_height)
    right = ceil(left + part_width)
    lower = ceil(upper + part_height)
    return image.crop((left, upper, right, lower)).convert('RGBA')


def _get_card_name_img(text, font_size=40, padding=10):
    # use default font
    font = ImageFont.load_default(size=font_size)    

    # determine textbox dimensions
    draw = ImageDraw.Draw(Image.new('RGBA', (1, 1)))
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # create transparent image and draw card name
    card_name_img = Image.new(
        'RGBA', (text_width + 2 * padding, text_height + 2 * padding),
        (0, 0, 0, 0))
    draw = ImageDraw.Draw(card_name_img)
    draw.text((padding, padding), text, font=font, fill=(255, 255, 255, 255))
    return card_name_img


def _get_card_effect_img(
        text, box_size, font_size=10, padding=10):
    width, height = box_size

    # use default font
    font = ImageFont.load_default(size=font_size)

    # estimate number of characters per line
    draw = ImageDraw.Draw(Image.new('RGBA', (1, 1)))
    avg_char_width = draw.textlength('A', font=font) * 0.1 + draw.textlength(
        'a', font=font) * 0.9
    max_chars_per_line = int(max(
        (width - 2 * padding) // avg_char_width, 1))

    # wrap the text
    wrapped_text = textwrap.fill(text, width=max_chars_per_line)

    # check if wrapped text can fit in the provided dimensions
    lines = wrapped_text.split('\n')
    line_height = font.getbbox('A')[3] - font.getbbox('A')[1]
    line_space = 1
    total_text_height = line_height * len(lines) + line_space * (
        len(lines) - 1)
    if total_text_height + 2 * padding > height:
        raise Exception('Provided text cannot fit in text box')

    # create transparent image and draw card effect
    img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    y = padding
    for line in lines:
        x = padding
        draw.text((x, y), line, font=font, fill=(0, 0, 0, 255))
        y += line_height + 1
    return img


class CardCrafter:
    def __init__(self, dataset_path='cardcrafter_dataset.csv',
                 lora_dir='./cardcrafter_lora', using_ampere_gpu=False):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # get logger
        self.logger = logging.getLogger('CardCrafter_Logger')
        self.logger.setLevel(logging.INFO)

        # load multimodal model from Hugging Face for generating card image
        # descriptions
        model_path = 'microsoft/Phi-4-multimodal-instruct'
        self.logger.info(f'Loading {model_path} model...')
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map=self.device, torch_dtype='auto',
            trust_remote_code=True,
            _attn_implementation='flash_attention_2'
            if (using_ampere_gpu and 'cuda' in str(self.device)) else 'eager'
            ).to(self.device)
        self.generation_config = GenerationConfig.from_pretrained(model_path)
        self.logger.info('Loaded model')

        # specify tokens used in prompt
        self.user_token = '<|user|>'
        self.assistant_token = '<|assistant|>'
        self.end_token = '<|end|>'

        # get diffusion model pipeline for generating image
        self.cc_img_gen = CardCrafterImageGenerator(lora_dir)

        # create few-shot prompt for image description generation
        dataset_df = pd.read_csv(dataset_path)
        trap_df = dataset_df[dataset_df['COMMON_Card_Type'] == 'Trap'].iloc[:4]
        spell_df = dataset_df[
            dataset_df['COMMON_Card_Type'] == 'Spell'].iloc[:4]
        few_shot_df = pd.concat(
            [spell_df, trap_df], axis=0, ignore_index=True)[
                ['COMMON_Card_Name', 'COMMON_Effect',
                 'COMMON_Image_Description']]
        self.few_shot_prompt = ''
        for _, row in few_shot_df.iterrows():
            name = row['COMMON_Card_Name']
            effect = row['COMMON_Effect']
            img_desc = row['COMMON_Image_Description']
            example = f'{self.user_token}{_create_prompt(name, effect)}'\
                + f'{self.end_token}{self.assistant_token}{img_desc}'\
                + f'{self.end_token}'
            self.few_shot_prompt += example
        self.logger.info(
            'Assembled few-shot learning prompt for generating image '
            + 'descriptions from card info')

    def generate(self, card_type, name, effect):
        # get image description
        self.logger.info(
            'Generating image description from provided card info...')
        prompt = f'{self.few_shot_prompt}{self.user_token}'\
            + f'{_create_prompt(name, effect)}{self.end_token}'\
            + f'{self.assistant_token}'
        inputs = self.processor(
            text=prompt, images=None, return_tensors='pt').to(self.device)
        generate_ids = self.model.generate(
            **inputs, max_new_tokens=75,
            generation_config=self.generation_config)
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0]
        start_idx = response.rfind(self.assistant_token) + len(
            self.assistant_token)
        img_desc = response[start_idx:].strip()
        self.logger.info('Generated image description')

        # generate image from description
        card_img = self.cc_img_gen.generate(img_desc)

        # get Yu-Gi-Oh! card template
        card = _get_card_template(card_type)

        # add image
        card_img = card_img.resize((252, 254)).convert('RGBA')
        card.paste(card_img, (49, 112), card_img)

        # add name
        card_name_img = _get_card_name_img(name, font_size=25, padding=10)
        card.paste(card_name_img, (27, 25), card_name_img)

        # add effect
        card_effect_img = _get_card_effect_img(effect, (290, 80), 10, 5)
        card.paste(card_effect_img, (35, 388), card_effect_img)

        self.logger.info('Generated Yu-Gi-Oh! card')

        # return generated card
        return card
