# BERT_classifer_trial
BERT trial for chinese corpus classfication


Please download pretrained BERT chinise model from [Pretrained Chinese Model](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)

Just run `launch.sh` then the fine-tune process will intial.

Please modify the path from `launch.sh` and the path of out_dir in the `run_classfier.py` script

If you want to do the inference, you need to prepare the test file and open do_predict in `run_classfier.py`.

Script `predict_eval.py` may help you evaluate the inference result.
