

# related directories and filenames


dataset_root_dir="${proj_dir}/datasets/tmp/"
output_root_dir="${proj_dir}/output/"

data_dir="${dataset_root_dir}/${data_name}/${task}/"
output_dir="${output_root_dir}/${data_name}/${task}/"
#res_dir="${output_dir}/results/"

if [ ! -d $dataset_root_dir ]; then
    mkdir -p $dataset_root_dir
fi

if [ ! -d $data_dir ]; then
    mkdir -p $data_dir
fi

if [ ! -d $output_dir ]; then
    mkdir -p $output_dir
fi

#if [ ! -d $res_dir ]; then
#    mkdir -p $res_dir
#fi



