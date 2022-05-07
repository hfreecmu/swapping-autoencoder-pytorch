import os
import torchvision.transforms as transforms
from PIL import Image
from evaluation import BaseEvaluator
from data.base_dataset import get_transform
import util
import torch
import numpy as np


class TextureExtractEvaluator(BaseEvaluator):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument("--method", required=True, type=str)


        #required for save and interpolate
        parser.add_argument("--latent_type", type=str)
        #required for save and save all
        parser.add_argument("--input_dir", type=str)
        #required for interpolate
        parser.add_argument("--input_structure_image", type=str)
        parser.add_argument("--texture_mix_alphas", type=float, nargs='+', default=[1.0])
        parser.add_argument("--latent_mix_alphas", type=float, nargs='+', default=[1.0])
        
        opt, _ = parser.parse_known_args()
        dataroot = opt.input_dir
        
        # dataroot and dataset_mode are ignored in SimpleSwapplingEvaluator.
        # Just set it to the directory that contains the input structure image.
        parser.set_defaults(dataroot=dataroot, dataset_mode="imagefolder")
        
        return parser
    
    def load_image(self, path):
        path = os.path.expanduser(path)
        img = Image.open(path).convert('RGB')
        transform = get_transform(self.opt)
        tensor = transform(img).unsqueeze(0)
        return tensor
    
    def evaluate(self, model, dataset, nsteps=None):
        if self.opt.method == 'save':
            return self.save(model, self.opt.input_dir, self.opt.latent_type)
        elif self.opt.method == 'interpolate':
            return self.interpolate(model)
        elif self.opt.method == 'save_all':
            return self.save_all(model)
        else:
            raise RuntimeError('Illegal method: ' + self.opt.method)
        
    def save_all(self, model):
        assert self.opt.input_dir is not None

        for dirname in os.listdir(self.opt.input_dir):
            dir_path = os.path.join(self.opt.input_dir, dirname)
            if not os.path.isdir(dir_path):
                continue
            
            latent_type = dirname
            _ = self.save(model, dir_path, latent_type)


        return {}

    def save(self, model, input_dir, latent_type):
        assert input_dir is not None
        assert latent_type is not None

        output_dir = os.path.join(self.output_dir(), latent_type)

        os.makedirs(output_dir, exist_ok=True)

        latent_codes = []
        for filename in os.listdir(input_dir):
            if not (filename.endswith('jpg') or filename.endswith('.png')):
                continue

            image_path = os.path.join(input_dir, filename)

            print('Processing: ' + image_path)
            image = self.load_image(image_path)
        
            #TODO do I need to do this once or every time?
            model(sample_image=image, command="fix_noise")

            _, texture_code = model(image, command="encode")

            latent_codes.append(texture_code)

        latent_codes = torch.cat(latent_codes, dim=0)
        
        torch.save(latent_codes, os.path.join(output_dir, 'latent_codes.pth'))

        return {}

    def interpolate(self, model):
        assert self.opt.input_structure_image is not None
        assert self.opt.latent_type is not None

        #latent_dir should have the latent_codes.pth
        latent_dir = os.path.join(self.output_dir(), self.opt.latent_type)
        latent_codes = torch.load(os.path.join(latent_dir, 'latent_codes.pth'))

        inds = np.random.choice(latent_codes.shape[0], size=(2), replace=False)
        ind_0 = inds[0]
        ind_1 = inds[1]

        latent_0 = latent_codes[ind_0:ind_0+1]
        latent_1 = latent_codes[ind_1:ind_1+1]

        structure_image = self.load_image(self.opt.input_structure_image)

        model(sample_image=structure_image, command="fix_noise")
        structure_code, source_texture_code = model(
            structure_image, command="encode")

        texture_alphas = self.opt.texture_mix_alphas
        latent_alphas = self.opt.latent_mix_alphas

        for texture_alpha in texture_alphas:
            for latent_alpha in latent_alphas:
                latent_code = util.lerp(
                    latent_0, latent_1, latent_alpha)

                texture_code = util.lerp(
                    source_texture_code, latent_code, texture_alpha)

                output_image = model(structure_code, texture_code, command="decode")
                output_image = transforms.ToPILImage()(
                    (output_image[0].clamp(-1.0, 1.0) + 1.0) * 0.5)

                output_name = "%s_%.2f_%.2f.png" % (
                    os.path.splitext(os.path.basename(self.opt.input_structure_image))[0],
                    texture_alpha, latent_alpha
                )

                output_path = os.path.join(latent_dir, output_name)

                output_image.save(output_path)
                print("Saved at " + output_path)

        return {}


