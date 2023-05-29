python train.py --data_dir /opt/ml/input/data/medical/img/total_img \
    --json_dir /opt/ml/input/data/medical/ufo/img300.json 

# # resume 파일 넣는 경우
# python train.py --data_dir /opt/ml/input/data/medical/img/total_img \
#     --json_dir /opt/ml/input/data/medical/ufo/img300.json \
#     --resume /opt/ml/input/code/trained_models/latest.pth

# # resume 파일과 이전 best_loss 값 넣는 경우
# python train.py --data_dir /opt/ml/input/data/medical/img/total_img \
#     --json_dir /opt/ml/input/data/medical/ufo/img300.json \
#     --resume /opt/ml/input/code/trained_models/latest.pth \
#     --best_loss 0.2527

