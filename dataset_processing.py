import ast
import kagglehub
import logging
import os
import pandas as pd
from PIL import Image
import re
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig


EXTRA_DECK_TYPES = ['fusion', 'synchro', 'xyz', 'link']
SEARCH_TERMS_DICT = {
    'synchro': [
        '-Type monsters', 'non-Tuner monsters', 'non-Tuner monster',
        '-Type monster', 'monsters ', 'monster '],
    'fusion': [
        '-Type monster', '-Type monsters', 'monsters ', 'monster '],
    'xyz': [f'Level {i} monsters' for i in range(1, 13)] + [
        '-Type monsters', 'monsters '],
    'link': ['Effect Monsters', 'monster ', 'monsters ']
}
MATERIALS_EXTRACTION_EXAMPLES = [
    # Stardust Dragon (Synchro monster)
    (
        '1 Tuner + 1+ non-Tuner monsters\nWhen a card or effect '
        + 'is activated that would destroy a card(s) on the '
        + 'field (Quick Effect): You can Tribute this card; '
        + 'negate the activation, and if you do, destroy it. '
        + 'During the End Phase, if this effect was activated '
        + 'this turn (and was not negated): You can Special '
        + 'Summon this card from your GY.',
        '1 Tuner 1+ non-Tuner monsters'
    ),
    # Gaia Saber, The Lightning Shadow (Link monster)
    (
        '2+ monsters',
        '2+ monsters'
    ),
    # Buster Blader, the Dragon Destroyer Swordsman (Fusion
    # monster)
    (
        # missing + sign between monster materials, missing newline
        # character between monster materials and effect
        '1 "Buster Blader" 1 Dragon monster Must be Fusion '
        + 'Summoned. Cannot attack directly. Gains 1000 ATK/DEF '
        + 'for each Dragon monster your opponent controls or is '
        + 'in their GY. Change all Dragon monsters your opponent '
        + 'controls to Defense Position, also Dragon monsters in '
        + 'your opponent\'s possession cannot activate their '
        + 'effects. If this card attacks a Defense Position '
        + 'monster, inflict piercing battle damage.',
        '1 "Buster Blader" 1 Dragon monster'
    ),
    # Grenosaurus (Xyz monster)
    (
        '2 Level 3 monsters\nWhen this card destroys an '
        + 'opponent\'s monster by battle and sends it to the '
        + 'Graveyard: You can detach 1 Xyz Material from this '
        + 'card; inflict 1000 damage to your opponent.',
        '2 Level 3 monsters'
    ),
    # Cyber Dragon Infinity (Xyz monster)
    (
        # missing newline character between monster materials
        '3 Level 6 LIGHT Machine monsters '
        + 'Once per turn, you can also Xyz Summon "Cyber Dragon '
        + 'Infinity" by using "Cyber Dragon Nova" you control as '
        + 'material. (Transfer its materials to this card.) Gains '
        + '200 ATK for each material attached to it. Once per '
        + 'turn: You can target 1 face-up Attack Position monster '
        + 'on the field; attach it to this card as material. Once '
        + 'per turn, when a card or effect is activated (Quick '
        + 'Effect): You can detach 1 material from this card; '
        + 'negate the activation, and if you do, destroy it.',
        '3 Level 6 LIGHT Machine monsters'
    ),
    # Underground Arachnid (Synchro monster)
    (
        # no newline character between materials and effect
        '1 DARK Tuner + 1 non-Tuner Insect-Type monster '
        + 'If this card attacks, your opponent cannot activate '
        + 'any Spell or Trap Cards until the end of the Damage '
        + 'Step. Once per turn, you can select 1 face-up monster '
        + 'your opponent controls, and equip it to this card. If '
        + 'this card would be destroyed by battle, you can '
        + 'destroy the monster equipped to this card by its '
        + 'effect instead. (You can only equip 1 monster at a '
        + 'time to this card.)',
        '1 DARK Tuner + 1 non-Tuner Insect-Type monster'
    ),
    # Blue Eyes Ultimate Dragon (Fusion monster)
    (
        '"Blue-Eyes White Dragon" + "Blue-Eyes White Dragon" + '
        + '"Blue-Eyes White Dragon"',
        '"Blue-Eyes White Dragon" + "Blue-Eyes White Dragon" + '
        + '"Blue-Eyes White Dragon"',
    ),
    # Decode Talker (Link monster)
    (
        '2+ Effect Monsters\nGains 500 ATK for each monster it '
        + 'points to. When your opponent activates a card or '
        + 'effect that targets a card(s) you control (Quick '
        + 'Effect): You can Tribute 1 monster this card points '
        + 'to; negate the activation, and if you do, destroy that '
        + 'card.',
        '2+ Effect Monsters'
    )
]


def _process_monster_img(img):
    return img.resize((900, 1314)).crop((60, 1025, 840, 1190))


def _process_pendulum_monster_imgs(img):
    return [
        img.resize((900, 1314)).crop((120, 825, 775, 980)),
        img.resize((900, 1314)).crop((60, 1025, 840, 1190))]


def _process_spell_trap_img(img):
    return img.resize((900, 1314)).crop((60, 988, 840, 1240))


def _split_with_search_terms(text, card_type):
    # get search terms
    for extra_deck_type in EXTRA_DECK_TYPES:
        if extra_deck_type in card_type:
            search_terms = SEARCH_TERMS_DICT[extra_deck_type]

    # find earliest occurance of any search term
    found_idx = None
    found_term = None
    for search_term in search_terms:
        if search_term in text:
            new_found_idx = text.find(search_term)
            if found_idx is None:
                found_idx = new_found_idx
                found_term = search_term
            else:
                if new_found_idx < found_idx or (
                        (new_found_idx == found_idx) and (
                        len(search_term) > len(found_term))):
                    found_idx = new_found_idx
                    found_term = search_term

    # split text accordingly
    if found_term is None:
        split_texts = ['', text]
    else:
        split_idx = found_idx + len(found_term)
        split_texts = [
            text[:split_idx].strip(),
            text[split_idx:].strip()
        ]
    return split_texts


def _remove_monster_types(text):
    text_shortened = True
    new_text = text
    while text_shortened:
        orig_len = len(new_text)
        new_text = re.sub(
            r'^(?:\[[^\]]*\]|\b[A-Z]+\b|/)+\s*', '', new_text)
        new_len = len(new_text)
        if orig_len == new_len:
            text_shortened = False
    return new_text


def _process_card_img(img, is_pendulum_monster):
    if is_pendulum_monster:
        crop_vals = (60, 235, 835, 810)
    else:
        crop_vals = (98, 235, 803, 938)
    return img.resize((900, 1314)).crop(crop_vals)


class DatasetProcessor:
    def __init__(self):
        # set device
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # get logger
        self.logger = logging.getLogger('CardCrafter_Logger')
        self.logger.setLevel(logging.INFO)

    def _load_dataset(self):
        # load Yu-Gi-Oh! card dataset from Kaggle
        dataset = 'archanghosh/yugioh-database'
        self.logger.info(
            f'Loading Yu-Gi-Oh card dataset from Kaggle ({dataset})...')
        self.kaggle_dataset_path = kagglehub.dataset_download(dataset)
        self.logger.info(f'Loaded dataset to {self.kaggle_dataset_path}')

    def _load_multimodal_model(self, using_ampere_gpu):
        # load multimodal model from Hugging Face
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

    def _generate_model_responses(
            self, prompts, imgs=None, max_new_tokens=1000):
        # pass inputs to model, get responses
        inputs = self.processor(
            text=prompts, images=imgs, return_tensors='pt').to(self.device)
        generate_ids = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            generation_config=self.generation_config)
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        responses = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)
        return responses

    def _get_imgs_from_filenames(self, img_filenames):
        # get images from file system
        imgs = []
        for img_filename in img_filenames:
            img_path = os.path.join(
                self.kaggle_dataset_path, 'Yugi_images', img_filename)
            img = Image.open(img_path)
            imgs.append(img)
        return imgs

    def _get_materials_and_effect(self, text, card_type):
        # assemble examples for few-shot learning
        prompt_prefix = 'Return the materials for this Yugioh card '\
            + 'exactly as they are mentioned in the following text: '
        prompt = ''
        for card_text, materials in MATERIALS_EXTRACTION_EXAMPLES:
            prompt += f'{self.user_token}{prompt_prefix}{card_text}'\
                + f'{self.end_token}{self.assistant_token}'\
                + f'{materials}{self.end_token}'

        # prepare prompt
        prompt += f'{self.user_token}{prompt_prefix}{text}'\
            + f'{self.end_token}{self.assistant_token}'

        # get materials and effect
        materials = self._generate_model_responses([prompt])[0]
        materials_idx = text.find(materials)
        if materials_idx == -1:
            # if language model does not work, use search terms
            materials, effect = _split_with_search_terms(text, card_type)
        else:
            effect = text[text.find(materials) + len(materials):]
        return materials.strip(), effect.strip()

    def _get_card_text(self, imgs, batch_df):
        def _update_text_dict(text_dict, text, card_type):
            # determine if card is an extra deck type
            is_extra_deck_type = False
            for extra_deck_type in EXTRA_DECK_TYPES:
                if extra_deck_type in card_type:
                    is_extra_deck_type = True

            subtext = _remove_monster_types(text)
            text_dict['Subtext'].append(subtext)
            if is_extra_deck_type:
                materials, effect = self._get_materials_and_effect(
                    subtext, card_type)
            else:
                materials = ''
                effect = subtext
            text_dict['Materials'].append(materials)
            text_dict['Effect'].append(effect)
            return text_dict

        # process images based on card type
        card_types = []
        processed_imgs = []
        for idx, row in batch_df.reset_index(drop=True).iterrows():
            card_type = row['Card type'].lower().strip()
            if card_type == 'monster':
                # check extra deck types
                ed_type = ''
                for extra_deck_type in EXTRA_DECK_TYPES:
                    if extra_deck_type in row['Types'].lower():
                        ed_type += f'{extra_deck_type} '
                        break

                # process images, track card types
                if 'pendulum' in row['Types'].lower():
                    card_types += [ed_type + 'pendulum ' + card_type] * 2
                    processed_imgs += _process_pendulum_monster_imgs(
                        imgs[idx])
                else:
                    card_types.append(ed_type + card_type)
                    processed_imgs.append(_process_monster_img(imgs[idx]))
            else:
                card_types.append(card_type)
                processed_imgs.append(_process_spell_trap_img(imgs[idx]))

        # get the text in the processed images
        prompt = f'{self.user_token}<|image_1|>What is the text in this ' \
            + 'image? Return the text exactly as it appears in the image.' \
            + f'{self.end_token}{self.assistant_token}'
        card_texts = self._generate_model_responses(
            [prompt] * len(processed_imgs), processed_imgs)
        # remove starting newline character from responses
        card_texts = list(map(lambda s: s.lstrip('\n'), card_texts))

        # return a dictionary with the categorized card text
        text_dict = {
            'Effect': [],
            'Pendulum_Effect': [],
            'Materials': [],
            'Subtext': []
        }
        i = 0
        while i < len(card_texts):
            # get pendulum effect, materials (for extra deck cards), and effect
            if 'pendulum' in card_types[i]:
                text_dict['Pendulum_Effect'].append(card_texts[i])
                text_dict = _update_text_dict(
                    text_dict, card_texts[i + 1], card_types[i])
                i += 2
            else:
                text_dict['Pendulum_Effect'].append('')
                text_dict = _update_text_dict(
                    text_dict, card_texts[i], card_types[i])
                i += 1
        return text_dict

    def _get_img_descs(self, imgs, is_pendulum_type):
        # get card image descriptions
        processed_imgs = list(map(
            lambda i: _process_card_img(imgs[i], is_pendulum_type[i]),
            range(len(imgs))))
        prompt = f'{self.user_token}<|image_1|>Describe this image in at '\
            + f'most 55 words. {self.end_token}{self.assistant_token}'
        img_descs = self._generate_model_responses(
            [prompt] * len(imgs), processed_imgs, max_new_tokens=75)
        return img_descs

    def _save_dataset_csv(self, dataset_df, save_path):
        # save updated dataset CSV file
        dataset_df.to_csv(save_path, index=False)
        self.logger.info(
            f'Saved updated dataset CSV file to {save_path}')

    def _update_columns(self, dataset_df):
        # process effect types column
        effect_types = []
        for _, row in dataset_df.iterrows():
            effect_types_val = row['Effect types']
            if pd.isna(effect_types_val) or len(effect_types_val) == 0:
                effect_types.append(None)
            else:
                effect_types.append(
                    ', '.join(ast.literal_eval(effect_types_val)))
        dataset_df['COMMON_Effect_Types'] = effect_types

        # process ATK/DEF and ATK/LINK columns
        atk_vals = []
        def_vals = []
        link_vals = []
        for _, row in dataset_df.iterrows():
            atk_def = row['ATK / DEF']
            atk_link = row['ATK / LINK']
            if not pd.isna(atk_def) and len(atk_def) != 0:
                vals = row['ATK / DEF'].split('/')
                atk_vals.append(vals[0].strip())
                def_vals.append(vals[1].strip())
                link_vals.append(None)
            elif not pd.isna(atk_link) and len(atk_link) != 0:
                vals = row['ATK / LINK'].split('/')
                atk_vals.append(vals[0].strip())
                def_vals.append(None)
                link_vals.append(vals[1].strip())
            else:
                atk_vals.append(None)
                def_vals.append(None)
                link_vals.append(None)
        dataset_df['MONSTER_Atk'] = atk_vals
        dataset_df['MONSTER_Def'] = def_vals
        dataset_df['MONSTER_Link'] = link_vals

        # process property column
        spell_prop_vals = []
        trap_prop_vals = []
        for _, row in dataset_df.iterrows():
            card_type = row['Card type'].lower().strip()
            prop_val = row['Property']
            if card_type == 'spell':
                spell_prop_vals.append(prop_val)
                trap_prop_vals.append(None)
            elif card_type == 'trap':
                trap_prop_vals.append(prop_val)
                spell_prop_vals.append(None)
            else:
                trap_prop_vals.append(None)
                spell_prop_vals.append(None)
        dataset_df['SPELL_Property'] = spell_prop_vals
        dataset_df['TRAP_Property'] = trap_prop_vals

        # process monster non-effect subtext column (only for Normal monsters)
        effect_vals = []
        non_effect_descs = []
        for _, row in dataset_df.iterrows():
            card_type = row['Card type'].lower().strip()
            effect_types = row['Effect types']
            no_effect = pd.isna(effect_types) or len(effect_types) == 0
            normal_type = not pd.isna(row['Types']) and (
                'normal' in row['Types'].lower().strip()
                or 'effect' not in row['Types'].lower().strip())
            if card_type == 'monster' and (no_effect or normal_type):
                # what was read by multi-modal model as the effect of the card
                # is the Normal monster's subtext
                non_effect_descs.append(row['Effect'])
                effect_vals.append(None)
            else:
                non_effect_descs.append(None)
                effect_vals.append(row['Effect'])
        dataset_df['COMMON_Effect'] = effect_vals
        dataset_df['MONSTER_Non_Effect_Description'] = non_effect_descs

        # process mosnter types columns
        dataset_df['MONSTER_Types'] = list(
            map(lambda v: f'[{v}]' if not pd.isna(v) and len(v) != 0 else v,
                dataset_df['Types']))

        # rename columns
        dataset_df = dataset_df.rename(columns={
            # common columns
            'Card type': 'COMMON_Card_Type',
            'Card_name': 'COMMON_Card_Name',
            'Image_Description': 'COMMON_Image_Description',
            'Rarity': 'COMMON_Rarity',
            'Image_name': 'COMMON_Image_Name',
            'Subtext': 'COMMON_Subtext',
            # monster columns
            'Attribute': 'MONSTER_Attribute',
            'Level': 'MONSTER_Level',
            'Rank': 'MONSTER_Rank',
            'Link Arrows': 'MONSTER_Link_Arrows',
            'Pendulum_Effect': 'MONSTER_Pendulum_Effect',
            'Pendulum Scale': 'MONSTER_Pendulum_Scale',
            'Ritual required': 'MONSTER_Summoned_By_Ritual_Spell',
            'Summoned by the effect of': 'MONSTER_Summoned_By_Effect_Of',
            'Materials': 'MONSTER_Materials',
            # spell card columns
            'Ritual Monster required': 'SPELL_Summons_Ritual_Monster'
        })

        # filter/order columns
        columns = [
            # common columns
            'COMMON_Card_Type',
            'COMMON_Card_Name',
            'COMMON_Image_Name',
            'COMMON_Image_Description',
            'COMMON_Rarity',
            'COMMON_Effect',
            'COMMON_Effect_Types',
            'COMMON_Subtext',
            # monster card columns
            'MONSTER_Non_Effect_Description',
            'MONSTER_Materials',
            'MONSTER_Pendulum_Effect',
            'MONSTER_Attribute',
            'MONSTER_Level',
            'MONSTER_Rank',
            'MONSTER_Link_Arrows',
            'MONSTER_Types',
            'MONSTER_Atk',
            'MONSTER_Def',
            'MONSTER_Link',
            'MONSTER_Pendulum_Scale',
            'MONSTER_Summoned_By_Ritual_Spell',
            'MONSTER_Summoned_By_Effect_Of',
            # spell card columns
            'SPELL_Property',
            'SPELL_Summons_Ritual_Monster',
            # trap card columns
            'TRAP_Property'
        ]
        dataset_df = dataset_df[columns]

        # return dataset DataFrame
        return dataset_df

    def process_dataset(
            self, batch_size, csv_save_path='./cardcrafter_dataset.csv',
            using_ampere_gpu=False):
        # load dataset
        self._load_dataset()

        # load multimodel model
        self._load_multimodal_model(using_ampere_gpu)

        # for each card in the dataset, get its image descriptions and effect/
        # subtext
        self.logger.info(
            'Adding image descriptions and card text to dataset...')
        dataset_df = pd.read_csv(
            os.path.join(self.kaggle_dataset_path, 'Yugi_db_cleaned.csv'))
        effects = []
        pendulum_effects = []
        materials = []
        img_descs = []
        subtext = []
        for i in tqdm(range(0, len(dataset_df), batch_size)):
            batch_df = dataset_df.iloc[i:i+batch_size]

            # get images
            imgs = self._get_imgs_from_filenames(batch_df['Image_name'])

            # get card effects
            card_text_dict = self._get_card_text(imgs, batch_df)
            effects += card_text_dict['Effect']
            pendulum_effects += card_text_dict['Pendulum_Effect']
            materials += card_text_dict['Materials']
            subtext += card_text_dict['Subtext']

            # clear cache to save memory
            torch.cuda.empty_cache()

            # get image descriptions
            is_pendulum_type = list(map(
                lambda t: not pd.isna(t) and 'pendulum' in t.lower(),
                batch_df['Types']))
            img_descs += self._get_img_descs(imgs, is_pendulum_type)

            # clear cache to save memory
            torch.cuda.empty_cache()
        dataset_df['Effect'] = effects
        dataset_df['Pendulum_Effect'] = pendulum_effects
        dataset_df['Image_Description'] = img_descs
        dataset_df['Materials'] = materials
        dataset_df['Subtext'] = subtext
        self.logger.info(
            'Added image descriptions and card effects to dataset')

        # update columns
        dataset_df = self._update_columns(dataset_df)

        # save updated dataset CSV
        self._save_dataset_csv(dataset_df, csv_save_path)
