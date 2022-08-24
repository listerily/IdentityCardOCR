########################################################
#
#    MODULE CARD GENERATOR
#      CARD GENERATOR generates fake images of ID card
#    which is used to test.
#
########################################################
import json
import os.path
import random
import matplotlib.pyplot as plt
from perlin_noise import PerlinNoise
import numpy as np
import cv2
import pandas as pd
from PIL import Image, ImageFont, ImageDraw


# Returns the num of days in a month
def get_maximum_days_num(year, month):
    month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if year % 4 == 0 and year % 400 != 0 and month == 2:
        return month_days[month - 1] + 1
    return month_days[month - 1]


# Make colors of an image brighter or darker based on brightness matrix
def compose_brightness(original, brightness):
    val = original + brightness * 0.35
    return 0 if val <= 0. else 1 if val >= 1. else val


# Randomly generate fake card number
def generate_card_number(code, gender, birth_year, birth_month, birth_day):
    # Generate first 17 card numbers
    birth_month_str = ('0' if birth_month < 10 else '') + str(birth_month)
    birth_day_str = ('0' if birth_day < 10 else '') + str(birth_day)
    birth_str = str(birth_year) + birth_month_str + birth_day_str
    identity_code = random.randint(0, 99)
    identity_code_str = ('0' if identity_code < 10 else '') + str(identity_code)
    identity_code_str += str(gender + random.randint(0, 4) * 2)  # Considered male and female difference
    body_str = str(code) + birth_str + identity_code_str
    # Calculate validation code
    weighted_sum = (np.array([int(x) for x in body_str]) * np.array([7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2])).sum()
    t = weighted_sum % 11
    r = (12 - t) % 11
    return body_str + ('X' if r == 10 else str(r))


class Generator:
    def __init__(self, count, seed=None):
        self.count = count
        # Specify random seed
        if seed is not None:
            random.seed(seed)

        # Read card background image
        self.background_image = cv2.imread('./res/background.png')
        self.background_image = cv2.cvtColor(self.background_image, cv2.COLOR_BGR2RGB)
        self.image_width, self.image_height = self.background_image.shape[0], self.background_image.shape[1]
        self.background_image = self.background_image.astype(float) / 255.

        # Read font
        self.font = ImageFont.truetype('./res/HuaWenHeiTi.ttf', 60)
        self.font_large = ImageFont.truetype('./res/HuaWenHeiTi.ttf', 70)
        self.font_number = ImageFont.truetype('./res/OCR-B_10_BT.ttf', 75)

        # Read person image
        self.person_male_image = Image.open('res/person-male.png')
        self.person_female_image = Image.open('res/person-female.png')
        self.person_male_image = self.person_male_image.resize((450, 530))
        self.person_female_image = self.person_female_image.resize((450, 530))

        # Read database
        self.names = pd.read_csv('./res/chinese_name.csv')
        self.nationalities = pd.read_csv('./res/chinese_nationality.csv')
        self.streets = pd.read_csv('res/chinese_street.csv')
        self.id_card_codes = pd.read_csv('res/id_card_code.csv')

    def generate(self, write=False, debug=True, add_noise=True, directory='./cards'):
        # Generation loop
        for i in range(self.count):
            self.generate_one_image(i, write, debug, add_noise, directory)

    def generate_one_image(self, idx, write=False, debug=True, add_noise=True, directory='./cards'):
        generated_image = np.copy(self.background_image)
        # Randomly generate identity data
        name = self.names.name[random.randint(0, len(self.names) - 1)]
        nationality = self.nationalities.name[random.randint(0, len(self.nationalities) - 1)] if random.randint(0, 11) == 0 else '汉'
        birth_year = 1952 + int(random.uniform(0, 1) * 70)
        birth_month = random.randint(1, 12)
        birth_day = random.randint(1, get_maximum_days_num(birth_year, birth_month))
        gender = random.randint(0, 1)
        code, address = self.id_card_codes.loc[random.randint(0, len(self.id_card_codes) - 1), 'code':'name']
        street = self.streets.name[random.randint(0, len(self.streets) - 1)]
        street_number = random.randint(1, 999)
        address += street + str(street_number) + '号'
        number_str = generate_card_number(code, gender, birth_year, birth_month, birth_day)

        # Export data to a dictionary
        generated_data = {
            'name': name,
            'nationality': nationality,
            'birth_day': birth_day,
            'birth_month': birth_month,
            'birth_year': birth_year,
            'address': address,
            'card_number': number_str,
            'gender': '男' if gender == 1 else '女',
        }

        # Draw text on the specific position of the IDCard image
        pil_image = Image.fromarray(np.uint8(generated_image.astype(float) * 255))
        image_draw = ImageDraw.Draw(pil_image)
        name_starting_x = 350
        for c in name:
            image_draw.text((name_starting_x, 190), c, (0, 0, 0, 0), font=self.font_large)
            name_starting_x += self.font_large.getlength(c) + 6
        image_draw.text((350, 328), '男' if gender == 1 else '女', (0, 0, 0, 0), font=self.font)
        image_draw.text((725, 328), nationality, (0, 0, 0, 0), font=self.font)
        image_draw.text((350, 463), str(birth_year), (0, 0, 0, 0), font=self.font)
        image_draw.text((630, 463), str(birth_month), (0, 0, 0, 0), font=self.font)
        image_draw.text((810, 463), str(birth_day), (0, 0, 0, 0), font=self.font)
        number_starting_x = 580
        for c in number_str:
            image_draw.text((number_starting_x, 920), c, (0, 0, 0, 0), font=self.font_number)
            number_starting_x += self.font_number.getlength(c) + 7
        address_starting_y = 590
        while len(address) > 0:
            address_starting_x = 350
            for c in address[:12]:
                image_draw.text((address_starting_x, address_starting_y), c, (0, 0, 0, 0), font=self.font)
                address_starting_x += self.font.getlength(c) + 2
            address = address[12:]
            address_starting_y += 80
        # Paste a profile photo to fake ID Card.
        if gender == 1:
            pil_image.paste(self.person_male_image, (1170, 250), mask=self.person_male_image)
        else:
            pil_image.paste(self.person_female_image, (1170, 250), mask=self.person_female_image)
        generated_image = np.array(pil_image) / 255.

        # Add noise
        if add_noise:
            # Generate perlin noise and apply brightness changes to this image.
            # Perlin Noise is a type of gradient noise.
            # Using perlin noise we could generate fake lights and shades on the IDCard.
            noise = PerlinNoise(octaves=0.8)
            generated_noise = [[noise([i / self.image_width, j / self.image_height])
                                for j in range(self.image_height)] for i in range(self.image_width)]
            generated_noise = np.stack((generated_noise, generated_noise, generated_noise), axis=-1)
            brightness_composer = np.frompyfunc(compose_brightness, 2, 1)
            generated_image = brightness_composer(generated_image, generated_noise)

        if debug:
            # If option debug turned on, this program would show images using matplotlib
            plt.imshow(generated_image.astype(float))
            plt.show()

        if write:
            # Write generated images to disk
            os.makedirs(directory, exist_ok=True)
            Image.fromarray(np.uint8(generated_image.astype(float) * 255)).save(directory + '/' + str(idx) + '.png')
            with open(directory + '/' + str(idx) + '.json', 'w', encoding='utf8') as f:
                json.dump(generated_data, f, ensure_ascii=False)
                f.close()
        return generated_image, generated_data


if __name__ == '__main__':
    Generator(1000).generate(debug=False, write=True)
