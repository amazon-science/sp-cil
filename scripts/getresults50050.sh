# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

set -e
branch="fc"
size="18"
branch="layer4"
size="10"
while getopts l:n: flag
do
    case "${flag}" in
        l) branch=${OPTARG};;
        n) size=${OPTARG};;
    esac
done

if [ "$branch" != "fc" ] && [ "$branch" != "layer4" ]; then
	echo 'not a valid argument in layer layer existing'
	exit
fi

if [ "$size" == "10" ]; then
  shots=""
  ep=10
elif [ "$size" == "18" ]; then
  shots="_20shot"  # Note ResNet18 is run with 20-shot to compare to prior work
  ep=5
else
	echo 'not a valid argument in network setting ' $config
	exit
fi
network=resnet$size

echo 'Creating dataset splits ....'
(cd prepro/ && python gen_split500.py)

echo 'Creating config scripts for stage-I training ....'
# base is already there and this will generate novel training files
(cd configs/multi_step && python createyaml.py --baseyml imnet_base_${network}_500.yml --size $size --branch $branch --outputdir='.')
echo 'Creating config scripts for stage-II fusion ....'
(cd configs/multi_step && python createyaml_fusion.py)
echo 'Creating config scripts for testing ....'
(cd configs/multi_step/test/${network} && python createyaml.py)

echo 'Starting base training for ' $network
python3 src/train_base.py --config configs/multi_step/imnet_base_${network}_500.yml

echo 'Starting novel training ....'
echo 'Note: these can run parallel.'
for step in 50 100 150 200 250 300 350 400 450 500; do
  python src/train_novel.py --config configs/multi_step/imnet_novel_${network}_${branch}_${step}.yml
done

echo 'Starting score fusion training ....'
echo 'Note: these can run parallel as soon as the corresponding novel training is finished.'
batch_size=512
for step in 1 2 3 4 5 6 7 8 9 10; do
  train_config=configs/multi_step/imnet_delta_${branch}_${network}${shots}_500_split${step}.yml
  test_config=configs/multi_step/test/${network}/deltacls_branches_split${step}.yaml
  python src/hyper_search.py --train_config ${train_config} --test_config ${test_config} --b ${batch_size} --script _ms --ep ${ep}
done

echo 'Aggregating step results...'
python scripts/collectresults50050.py ${network} ${branch}