from .tmux_launcher import Options, TmuxLauncher


class Launcher(TmuxLauncher):
    def options(self):
        opt = Options()
        opt.set(
            dataroot="~/datasets/flickr/mountain/train",
            dataset_mode="imagefolder",
            checkpoints_dir="./checkpoints/",
            num_gpus=7, batch_size=14,
            # scale the image such that the short side is |load_size|, and
            # crop a square window of |crop_size|.
            preprocess="scale_shortside_and_crop",
            load_size=512, crop_size=512,
        )

        return [
            opt.specify(
                name="mountain_pretrained",
                lambda_patch_R1=10.0,
            ),
        ]

    def train_options(self):
        common_options = self.options()
        return [opt.specify(
            display_freq=1600, print_freq=480,
            continue_train=True,
            evaluation_metrics="none") for opt in common_options]
        
    def test_options(self):
        opt = self.options()[0]
        return [
            # Swapping Grid Visualization. Fig 12 of the arxiv paper
            opt.tag("swapping_grid").specify(
                num_gpus=1,
                batch_size=1,
                #dataroot="./testphotos/mountain/fig12/",
                #dataset_mode="imagefolder",
                #preprocess="scale_shortside",  # For testing, scale but don't crop
                dataroot="~/datasets/testphotos/images_used_in_swapping_autoencoder_arxiv/mountain/fig12/",
                dataset_mode="imagefolder",
                preprocess="scale_width",  # For testing, scale but don't crop
                load_size=1024, crop_size=1024,
                evaluation_metrics="structure_style_grid_generation"
            ),
            
            # Simple Swapping code for quick testing
            opt.tag("simple_swapping").specify(
                num_gpus=1,
                batch_size=1,
                dataroot=".",  # dataroot is ignored.
                result_dir="./results/",
                preprocess="scale_shortside",
                load_size=512,
                evaluation_metrics="simple_swapping",
                # Specify the two images here.
                #input_structure_image="./testphotos/mountain/fig12/structure/AdobeStock_104191871.jpeg",
                #input_texture_image="./testphotos/mountain/fig12/style/AdobeStock_312564332.jpeg",
                input_structure_image='/home/frc-ag-3/harry_ws/visual_synthesis/final_project/data/dep/photos_mini/forest/forest_mix/000000.jpg',
                input_texture_image='/home/frc-ag-3/harry_ws/visual_synthesis/final_project/data/dep/photos_mini/mountains/winter_mountains/000002.jpg',
                # alpha == 1.0 corresponds to full swapping.
                # 0 < alpha < 1 means interpolation
                texture_mix_alpha=1.0,
            ),
            
            # Simple interpolation images for quick testing
            opt.tag("simple_interpolation").specify(
                num_gpus=1,
                batch_size=1,
                dataroot=".",  # dataroot is ignored.
                result_dir="./results/",
                #preprocess="scale_shortside",
                preprocess="resize",
                load_size=512,
                evaluation_metrics="simple_swapping",
                # Specify the two images here.
                #input_structure_image="./testphotos/mountain/fig12/structure/AdobeStock_104191871.jpeg",
                #input_texture_image="./testphotos/mountain/fig12/style/AdobeStock_312564332.jpeg",
                input_structure_image='/home/frc-ag-3/harry_ws/visual_synthesis/repos/swapping-autoencoder-pytorch/datasets/flickr_data/structures/google_highres/forest/forest1.png',
                input_texture_image='/home/frc-ag-3/harry_ws/visual_synthesis/repos/swapping-autoencoder-pytorch/datasets/flickr_data/latent_textures/snow/snow_000031.png',
                #texture_mix_alpha='0.0 0.25 0.5 0.75 1.0',
                texture_mix_alpha='0.0 0.25 0.5 0.75 0.9 1.0',
            ),

            # Extract all latent codes from directory
            opt.tag("extract_textures").specify(
                num_gpus=1,
                batch_size=1,
                dataroot=".",  # dataroot is ignored.
                result_dir="./results/",
                preprocess="scale_shortside",
                load_size=512,
                evaluation_metrics="texture_extract",
                method='save',
                latent_type='snow',
                input_dir='/home/frc-ag-3/harry_ws/visual_synthesis/final_project/data/photos/snow',
            ),

            # Extract all latent codes from multiple directories
            opt.tag("extract_all_textures").specify(
                num_gpus=1,
                batch_size=1,
                dataroot=".",  # dataroot is ignored.
                result_dir="./results/",
                preprocess="scale_shortside",
                #preprocess="resize",
                load_size=512,
                evaluation_metrics="texture_extract",
                method='save_all',
                input_dir='/home/frc-ag-3/harry_ws/visual_synthesis/final_project/data/flickr/latent_textures',
            ),

            # Extract all latent codes from directory
            opt.tag("texture_mix").specify(
                num_gpus=1,
                batch_size=1,
                dataroot=".",  # dataroot is ignored.
                result_dir="./results/",
                #preprocess="scale_shortside",
                preprocess="resize",
                load_size=512,
                evaluation_metrics="texture_extract",
                method='interpolate',
                latent_type='snow',
                input_structure_image='/home/frc-ag-3/harry_ws/visual_synthesis/final_project/data/flickr/structures/regular_dataset/forest_trees/forest_trees_000004.png',
                texture_mix_alpha=1.0,
                latent_mix_alpha='0.0 0.25 0.5 0.75 1.0',
            )
        ]
