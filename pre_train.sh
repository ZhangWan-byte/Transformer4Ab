declare -a arr=('masonscnn' 'lstm' 'textcnn' 'ag_fast_parapred' 'pipr' 'resppi' 'pesi')

for model_name in "${arr[@]}"
do
    echo "nohup python pre_train.py $model_name > ./logs/log_pretrain_$model_name 2>&1 &"

    nohup python pre_train.py $model_name > ./logs/log_pretrain_$model_name 2>&1 &
done