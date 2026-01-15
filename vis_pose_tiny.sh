python visualize_pose_predictions.py \
  --checkpoint /workspace/LWDETR/data/weights/checkpoint0029_new.pth \
  --split test \
  --dataset_path /workspace/LWDETR/data/datasets/bop/ycbv \
  --dataset_file ycbv \
  --encoder vit_tiny \
  --vit_encoder_num_layers 6 \
  --window_block_indexes 0 2 4 \
  --out_feature_indexes 1 3 5 \
  --n_mesh_points 1024 \
  --dec_layers 3 --group_detr 13 --two_stage \
  --projector_scale P4 --hidden_dim 256 \
  --sa_nheads 8 --ca_nheads 16 --dec_n_points 2 \
  --bbox_reparam --lite_refpoint_refine --num_queries 100 \
  --ia_bce_loss --cls_loss_coef 1 --num_select 100 \
  --square_resize_div_64 --use_ema \
  --pretrain_keys_modify_to_load transformer.enc_out_class_embed.0.weight transformer.enc_out_class_embed.1.weight transformer.enc_out_class_embed.2.weight transformer.enc_out_class_embed.3.weight transformer.enc_out_class_embed.4.weight transformer.enc_out_class_embed.5.weight transformer.enc_out_class_embed.6.weight transformer.enc_out_class_embed.7.weight transformer.enc_out_class_embed.8.weight transformer.enc_out_class_embed.9.weight transformer.enc_out_class_embed.10.weight transformer.enc_out_class_embed.11.weight transformer.enc_out_class_embed.12.weight transformer.enc_out_class_embed.0.bias transformer.enc_out_class_embed.1.bias transformer.enc_out_class_embed.2.bias transformer.enc_out_class_embed.3.bias transformer.enc_out_class_embed.4.bias transformer.enc_out_class_embed.5.bias transformer.enc_out_class_embed.6.bias transformer.enc_out_class_embed.7.bias transformer.enc_out_class_embed.8.bias transformer.enc_out_class_embed.9.bias transformer.enc_out_class_embed.10.bias transformer.enc_out_class_embed.11.bias transformer.enc_out_class_embed.12.bias class_embed.weight class_embed.bias \
  --grayscale \
  --pretrained_encoder /workspace/LWDETR/data/weights/caev2_tiny_S_300e_objects365.pth \
  --matcher_type hungarian --keypoint_loss_coef 1.0 \
  --trans_z_loss_coef 1.0 --trans_xy_loss_coef 1.0 \
  --rot_loss_coef 1.0 --adds_loss_coef 1.0 \
  --num_images 16 --score_threshold 0.4 --vis_output output/vis_test

  # --pretrain_weights /workspace/LWDETR/data/weights/LWDETR_xlarge_30e_objects365.pth \
