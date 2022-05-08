declare -a keywords=("dummy")
#declare -a keywords=("data a" "data b" "data c" "data d")
flickr_cred_name=harry
data_dir=/home/frc-ag-3/harry_ws/visual_synthesis/final_project/data/flickr/
num_images=100

for keyword in "${keywords[@]}"
do
   dir_keyword="${keyword// /_}"
   output_dir="$data_dir/$dir_keyword"
   mkdir -p $output_dir

   echo "Downloading $keyword to $output_dir" 

   python3 load_flickr.py --method "download_images" --flickr_cred_name $flickr_cred_name --keyword "$keyword" --output_dir $output_dir --num_images $num_images
done