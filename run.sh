python main.py --epoch 200 --tr_data_path ./dataSet/facades/train --transfer_type B_to_A --model_name facades
#python main.py --epoch 200 --tr_data_path ./dataSet/maps/train --model_name aerial2map
#python main.py --epoch 200 --tr_data_path ./dataSet/maps/train --model_name map2aerial --transfer_type B_to_A
#python main.py --epoch 15 --tr_data_path ./dataSet/edges2shoes/train --model_name edges2shoes --load_size 256 --batch_size 4 --in_memory False --mirroring False --random_jitter False

#python main.py --mode test --val_data_path ./dataSet/facades/val --transfer_type B_to_A --pre_trained_model ./model/pix2pix_facades --model_name facades --load_size 256
#python main.py --mode test --val_data_path ./dataSet/maps/val --pre_trained_model ./model/pix2pix_aerial2map --model_name aerial2map --load_size 256 --in_memory False
#python main.py --mode test --val_data_path ./dataSet/maps/val --pre_trained_model ./model/pix2pix_map2aerial --model_name map2aerial --load_size 256 --in_memory False --transfer_type B_to_A
#python main.py --mode test --val_data_path ./dataSet/edges2shoes/val --pre_trained_model ./model/pix2pix_edges2shoes --load_size 256 --model_name edges2shoes

