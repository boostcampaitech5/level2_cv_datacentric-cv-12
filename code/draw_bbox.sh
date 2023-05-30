# python draw_bbox.py --img_dir /opt/ml/input/data/medical/img/camper_img \
# 	--json_dir /opt/ml/input/data/medical/ufo/camper_img.json \
# 	--result_dir /opt/ml/input/data/drawed_bbox/camper_result

# inference 결과 확인 img300
python draw_bbox.py --img_dir /opt/ml/input/data/medical/img/test \
	--json_dir /opt/ml/input/code/predictions/img300_epoch199.json \
	--result_dir /opt/ml/input/data/drawed_bbox/img300_199_result

# # inference 결과 확인 100
# python draw_bbox.py --img_dir /opt/ml/input/data/medical/img/test \
# 	--json_dir /opt/ml/input/code/predictions/base_consinelr_adamw_weightdecay0.01_epoch200.json \
# 	--result_dir /opt/ml/input/data/drawed_bbox/base_weightdecay0.01_epoch200

