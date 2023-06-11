source_list=("../original_data/data/test.SRC" "../original_data/similar_lang/devtest/awa_Deva.devtest" 
"../original_data/similar_lang/devtest/bho_Deva.devtest" 
"../original_data/similar_lang/devtest/hne_Deva.devtest" 
"../original_data/similar_lang/devtest/mag_Deva.devtest"
"../original_data/similar_lang/devtest/mai_Deva.devtest" 
"../original_data/similar_lang/devtest/npi_Deva.devtest" 
"../original_data/similar_lang/devtest/san_Deva.devtest")


source_list=("../original_data/data/test.SRC" "../original_data/similar_lang/devtest/awa_Deva.devtest" 
"../original_data/similar_lang/devtest/bho_Deva.devtest" 
"../original_data/similar_lang/devtest/hne_Deva.devtest" 
"../original_data/similar_lang/devtest/mag_Deva.devtest"
"../original_data/similar_lang/devtest/mai_Deva.devtest" 
"../original_data/similar_lang/devtest/npi_Deva.devtest" 
"../original_data/similar_lang/devtest/san_Deva.devtest")

translated_list=("/hi_en/test.SRC" "/awa_Deva/awa_Deva.devtest" "/bho_Deva/bho_Deva.devtest" "/hne_Deva/hne_Deva.devtest" "/mag_Deva/mag_Deva.devtest" "/mai_Deva/mai_Deva.devtest" "/npi_Deva/npi_Deva.devtest" "/san_Deva/san_Deva.devtest")

reference_list=("../original_data/data/test.TGT" "../original_data/similar_lang/devtest/eng_Latn.devtest" 
"../original_data/similar_lang/devtest/eng_Latn.devtest" 
"../original_data/similar_lang/devtest/eng_Latn.devtest" 
"../original_data/similar_lang/devtest/eng_Latn.devtest" 
"../original_data/similar_lang/devtest/eng_Latn.devtest" 
"../original_data/similar_lang/devtest/eng_Latn.devtest" 
"../original_data/similar_lang/devtest/eng_Latn.devtest")

translated_folder=$1
device_id=$2

for i in 0 1 2 3 4 5 6 7
do 
folder_name=$(dirname ${translated_list[$i]})
mkdir -p $translated_folder/$folder_name
echo "========================================================"
echo "translating ${source_list[$i]}"
python inference.py --input_file ${source_list[$i]} \
    --output_file $translated_folder/${translated_list[$i]} \
    --checkpoint_path ../exp_2_1M/checkpoint-15620\
    --batch_size 32 \
    --device_id $device_id \
    --mbart_model_name "facebook/mbart-large-50-many-to-one-mmt"

echo "calculating BLEU"
sacrebleu ${reference_list[$i]} < $translated_folder/${translated_list[$i]} >> $translated_folder/${translated_list[$i]}.score -m bleu chrf
done

source ~/anaconda3/etc/profile.d/conda.sh
conda activate tf2

CUDA_VISIBLE_DEVICE=$device_id

for i in 0 1 2 3 4 5 6 7
do 
folder_name=$(dirname ${translated_list[$i]})
mkdir -p $translated_folder/$folder_name
echo "========================================================"

echo "calculating COMET"
command="comet-score -s ${source_list[$i]} -t $translated_folder/${translated_list[$i]} -r ${reference_list[$i]} --to_json $translated_folder/$folder_name/comet_calcualtions.json --quiet --only_system "
$command 2>&1 | tee $translated_folder/$folder_name/comet.log
echo "========================================================"

done

echo "calculating BLEURT"
python3 calculate_bleurt.py -f $translated_folder 2>&1 | tee $translated_folder/bleurt_score.log