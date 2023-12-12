34 23 * * * cd /root/grv_model/db_update
35 23 * * * /root/anaconda3/envs/pytorch/bin/python3 /root/grv_model/db_update/betfair_bfsp_updater.py > /root/grv_model/db_update/bf_update_cronlog.log 2>&1
40 23 * * * /root/anaconda3/envs/pytorch/bin/python3 /root/grv_model/db_update/database_update.py > /root/grv_model/db_update/db_update_cronlog.log 2>&1
43 23 * * * /root/anaconda3/envs/pytorch/bin/python3 /root/grv_model/db_update/database_update.py v6 > /root/grv_model/db_update/db_update_cronlog.log 2>&1
45 23 * * * cd /root/grv_model/model_predictions
46 23 * * * /root/anaconda3/envs/pytorch/bin/python3 /root/grv_model/model_predictions/update_model.py > /root/grv_model/model_predictions/cronlog.log 2>&1
50 23 * * * /root/anaconda3/envs/pytorch/bin/python3 /root/grv_model/model_predictions/update_model.py v6 > /root/grv_model/model_predictions/cronlog.log 2>&1
10 9 * * * cd /root/grv_model/db_update
11 9 * * * /root/anaconda3/envs/pytorch/bin/python3 /root/grv_model/db_update/new_form_gen.py > /root/grv_model/db_update/new_gen_cronlog.log 2>&1
11 9 * * * /root/anaconda3/envs/pytorch/bin/python3 /root/grv_model/db_update/new_form_gen.py v6 > /root/grv_model/db_update/new_gen_cronlog.log 2>&1
10 9 * * * cd /root/grv_model/model_predictions
11 9 * * * /root/anaconda3/envs/pytorch/bin/python3 /root/grv_model/model_predictions/update_model.py > /root/grv_model/model_predictions/cronlog.log 2>&1
15 9 * * * /root/anaconda3/envs/pytorch/bin/python3 /root/grv_model/model_predictions/generate_preds.py > /root/grv_model/model_predictions/cronlog.log 2>&1
20 11 * * * /root/anaconda3/envs/pytorch/bin/python3 /root/grv_model/flumine_betting/flumine_betting.py > /root/grv_model/flumine_betting/nz_cronlog.log 2>&1
10 11 * * * cd /root/grv_model/db_update
10 11 * * * cd /root/grv_model/db_update
40 8 * * * /root/anaconda3/envs/pytorch/bin/python3 /root/grv_model/db_update/new_form_gen.py v6 > /root/grv_model/db_update/new_gen_cronlog.log 2>&1
10 11 * * * cd /root/grv_model/model_predictions
11 11 * * * /root/anaconda3/envs/pytorch/bin/python3 /root/grv_model/model_predictions/update_model.py v6 > /root/grv_model/model_predictions/cronlog.log 2>&1
15 11 * * * /root/anaconda3/envs/pytorch/bin/python3 /root/grv_model/model_predictions/generate_preds.py v6 > /root/grv_model/model_predictions/cronlog.log 2>&1
20 11 * * * cd /root/grv_model/flumine_betting 
20 11 * * * /root/anaconda3/envs/pytorch/bin/python3 /root/grv_model/flumine_betting/relu_flumine_betting.py > /root/grv_model/flumine_betting/relu_cronlog.log 2>&1