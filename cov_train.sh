declare -a arr=('masonscnn' 'lstm' 'textcnn' 'ag_fast_parapred' 'pipr' 'resppi' 'pesi')

for modelname in "${arr[@]}"
do
    echo "python cov_train.py $modelname $1 > ./logs/${modelname}_$1 2>&1;"

    python cov_train.py $modelname $1 > ./logs/${modelname}_$1 2>&1;
done