from inception_v4 import create_inception_v4
import tflearn
import os,sys
X,Y = tflearn.data_utils.build_image_dataset_from_dir(os.path.join("/home/julyedu_51217/nn/data/dogandcats/", 'train/'),
        dataset_file="googlenet.pkl",
        resize=[227,227],
        filetypes=['.jpg', '.jpeg'],
        convert_gray=False,
        shuffle_data=True,categorical_Y=False)

model = create_inception_v4()
model.fit(X, Y)
