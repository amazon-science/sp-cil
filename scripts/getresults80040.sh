# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

set -e
branch="layer4"
size="10"
while getopts l:n: flag
do
    case "${flag}" in
        l) branch=${OPTARG};;
        n) size=${OPTARG};;
    esac
done
if [ "$branch" != "fc" ] && [ "$branch" != "layer4" ] 
then
	echo 'not a valid argument in layer layer existing'
	exit
fi
if [ "$size" != "10" ] && [ "$size" != "18" ]
then
	echo 'not a valid argument in network setting ' $config
	exit
fi
network=resnet$size

echo 'Creating dataset splits ....'
(cd prepro/ && python gen_split800_40_40.py)

echo 'Creating config scripts ....'
python3 scripts/createyaml.py --baseyml configs/table1/imnet_base_${network}.yml --size $size --branch $branch --outputdir=configs/table1

echo 'Starting base training for' $network
python3 src/train_base.py --config configs/table1/imnet_base_${network}.yml

echo 'Starting novel training ....'
python src/train_novel.py --config configs/table1/imnet_novel_${network}_${branch}.yml

echo 'Starting score fusion training ....'
train_config=configs/table1/imnet_delta_${network}_${branch}.yml
test_config=configs/table1/test_imnet_delta_${network}_${branch}.yml
batch_size=1024 
python src/hyper_search.py --train_config ${train_config} --test_config ${test_config} --b ${batch_size}
